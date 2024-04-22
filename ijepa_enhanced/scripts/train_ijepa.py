import copy
from typing import Optional
import torch
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, OmegaConf
import accelerate
from torch.utils.data import DataLoader
import wandb

from tensorsequence import TensorSequence

from ..vit import ViT
from ..ema import EMA
from ..predictor import Predictor
from ..lfq import LFQ, calculate_perplexity, masked_mean
from ..patchnpack import (
    MASK_IMAGE_ID,
    ContextTargetPatchNPacker,
    get_attention_mask,
)
from ..eval import eval_classification_probe
from ..utils import print_num_parameters
from ..dataset import get_dataset
from ..optimizer import get_optimizer


def compute_target_states(teacher: EMA, lfq: LFQ, tgt: TensorSequence, accelerator):
    # u8 to float
    tgt_patches = tgt["patches"] / 255
    tgt_image_ids = tgt["image_ids"]
    tgt_positions = tgt["positions"]
    tgt_attn_mask = get_attention_mask(tgt_image_ids)

    with torch.no_grad():
        with accelerator.autocast():
            tgt_states = teacher(tgt_patches, tgt_attn_mask, tgt_positions)
            lfq_result = lfq(tgt_states, return_indices=True, return_dict=True)
            lfq_states = lfq_result["hidden_states"]
            tgt_ids = lfq_result["indices"]

    return TensorSequence(
        named_columns={
            "ids": tgt_ids,
            "states": lfq_states,
            "positions": tgt_positions,
            "image_ids": tgt_image_ids,
        },
        sequence_dim=1,
    )


def compute_context_states(vit: ViT, lfq: LFQ, ctx: TensorSequence, accelerator):
    # Compute the context hidden states by using the vit with gradients enabled
    # u8 to float
    ctx_patches = ctx.named_columns["patches"] / 255
    ctx_positions = ctx.named_columns["positions"]
    ctx_image_ids = ctx.named_columns["image_ids"]
    ctx_attn_mask = get_attention_mask(ctx_image_ids)

    with accelerator.autocast():
        ctx_states = vit(ctx_patches, ctx_attn_mask, ctx_positions)
        lfq_result = lfq(
            ctx_states,
            mask=ctx_image_ids != MASK_IMAGE_ID,
            return_dict=True,
            return_losses=True,
        )
        ctx_states = lfq_result["hidden_states"]
        commit_loss = lfq_result["commit_loss"]
        entropy_loss = lfq_result["entropy_loss"]

    return (
        TensorSequence(
            named_columns={
                "states": ctx_states,
                "positions": ctx_positions,
                "image_ids": ctx_image_ids,
            },
            sequence_dim=1,
        ),
        commit_loss,
        entropy_loss,
    )


def compute_prediction_loss(
    ctx,
    tgt,
    predictor,
    prediction_block_masks,
    patchnpacker: ContextTargetPatchNPacker,
    accelerator,
):
    # Compute the loss from each target block mask and take the mean
    # For each block, the target states to predict are concatenated with the context states

    all_loss = []

    for prediction_block_mask in prediction_block_masks.unbind(-1):
        preds = patchnpacker.pack_prediction_target_sequence(
            tgt, ctx, prediction_block_mask
        )

        pred_ids = preds["ids"]
        pred_states = preds["states"]
        pred_positions = preds["positions"]
        pred_image_ids = preds["image_ids"]
        pred_tgt_mask = preds["prediction_mask"]
        # redo attention mask
        pred_attn_mask = get_attention_mask(pred_image_ids)

        with accelerator.autocast():
            y = predictor(pred_states, pred_attn_mask, pred_tgt_mask, pred_positions)

        # masked CE loss
        pred_ids.masked_fill_(~pred_tgt_mask.unsqueeze(-1), MASK_IMAGE_ID)
        num_classes = predictor.projection_dim
        loss = F.cross_entropy(
            y.view(-1, num_classes),
            pred_ids.view(-1),
            ignore_index=MASK_IMAGE_ID,
        )
        all_loss.append(loss)

    loss = torch.stack(all_loss).mean()
    return loss


def compute_training_losses(
    vit: ViT,
    lfq: LFQ,
    teacher: EMA,
    predictor: Predictor,
    accelerator,
    ctx: TensorSequence,
    tgt: TensorSequence,
    patchnpacker: ContextTargetPatchNPacker,
):
    device = accelerator.device
    # Compute target hidden states by passing the target patches through the teacher network

    prediction_block_masks = tgt["prediction_block_masks"]

    tgt = tgt.to_device(device)
    tgt = compute_target_states(teacher, lfq, tgt, accelerator)

    ctx = ctx.to_device(device)
    ctx, commit_loss, entropy_loss = compute_context_states(vit, lfq, ctx, accelerator)

    tgt_batch_size, tgt_sequence_length = tgt.all_columns[0].shape[:2]
    tgt_pred_mask = torch.ones(
        tgt_batch_size, tgt_sequence_length, device=device, dtype=torch.bool
    )
    tgt.named_columns["prediction_mask"] = tgt_pred_mask

    ctx_batch_size, ctx_sequence_length = ctx.leading_shape[:2]

    # building block for masking out preds
    # pred_mask is 1 where the patch is a prediction instead of a context
    ctx_pred_mask = torch.zeros(
        ctx_batch_size, ctx_sequence_length, device=device, dtype=torch.bool
    )

    ctx_ids = torch.full(
        (ctx_batch_size, ctx_sequence_length, lfq.num_codebooks),
        MASK_IMAGE_ID,
        device=device,
        dtype=torch.long,
    )

    ctx.named_columns["prediction_mask"] = ctx_pred_mask
    ctx.named_columns["ids"] = ctx_ids

    prediction_loss = compute_prediction_loss(
        ctx, tgt, predictor, prediction_block_masks, patchnpacker, accelerator
    )

    perplexity = calculate_perplexity(
        tgt["ids"], predictor.projection_dim, MASK_IMAGE_ID
    )

    return {
        "prediction_loss": prediction_loss,
        "commit_loss": commit_loss,
        "entropy_loss": entropy_loss,
        "perplexity": perplexity,
    }


@hydra.main(version_base=None, config_path="../../conf", config_name="conf")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    wandb.init(
        name="ijepa-enhanced",
        config=config,
        mode=config.wandb_mode,
    )

    vit = ViT(**config.model.vit)
    lfq = LFQ(**config.model.lfq)
    predictor = Predictor(**config.model.predictor)

    teacher = EMA(vit, **config.train.ema)
    print("vit: ", end="")
    print_num_parameters(vit)
    print("predictor: ", end="")
    print_num_parameters(predictor)
    print("lfq: ", end="")
    print_num_parameters(lfq)

    if config.torch_compile:
        vit.forward = torch.compile(vit.forward)
        teacher.forward = torch.compile(teacher.forward)
        predictor.forward = torch.compile(predictor.forward)
        lfq.forward = torch.compile(lfq.forward)

    dataset = get_dataset(**config.train.dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=config.num_workers,
    )

    patchnpacker = ContextTargetPatchNPacker(
        patch_size=config.model.vit.patch_size,
        batch_size=config.train.batch_size,
        sequence_length_context=config.train.sequence_length_context,
        sequence_length_target=config.train.sequence_length_target,
        sequence_length_prediction=config.train.sequence_length_prediction,
    )

    optimizer = get_optimizer(
        config.train.optimizer, list(vit.parameters()) + list(predictor.parameters())
    )

    accelerator = accelerate.Accelerator()
    device = accelerator.device

    vit, predictor, teacher, lfq, optimizer = accelerator.prepare(
        vit, predictor, teacher, lfq, optimizer
    )

    step = 0

    dataloader = iter(dataloader)

    for ctx, tgt in patchnpacker.make_iter(dataloader):
        optimizer.zero_grad()

        ctx.to_device(device)
        tgt.to_device(device)

        loss_dict = compute_training_losses(
            vit,
            lfq,
            teacher,
            predictor,
            accelerator,
            ctx,
            tgt,
            patchnpacker,
        )

        loss = (
            config.train.commit_loss_weight * loss_dict["commit_loss"]
            + config.train.entropy_loss_weight * loss_dict["entropy_loss"]
            + loss_dict["prediction_loss"]
        )

        accelerator.backward(loss)
        optimizer.step()

        teacher.update()

        loss_stmt = " ".join([f"{k}:{v.item():.5f}" for k, v in loss_dict.items()])

        print(f"train loss: {loss.item():.5f} {loss_stmt} step {step}")
        wandb.log({"train": {"loss": loss}}, step=step)

        del loss_dict
        del ctx, tgt

        if (step + 1) % config.train.eval_every_num_steps == 0:
            loss = eval_classification_probe(
                vit, lfq, copy.deepcopy(predictor), config.eval, None, accelerator
            )
            wandb.log({"eval": {"loss": loss}})
            vit.train()
            predictor.train()

        step += 1


if __name__ == "__main__":
    main()
