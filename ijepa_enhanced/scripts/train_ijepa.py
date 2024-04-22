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
from ..lfq import LFQ, masked_mean
from ..patchnpack import (
    MASK_IMAGE_ID,
    ContextTargetPatchNPacker,
    get_attention_mask,
)
from ..eval import eval_classification_probe
from ..utils import print_num_parameters
from ..dataset import get_dataset
from ..optimizer import get_optimizer


def compute_training_loss(
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
    tgt_patches = tgt.named_columns["patches"]
    tgt_positions = tgt.named_columns["positions"]
    tgt_image_ids = tgt.named_columns["image_ids"]
    tgt_block_masks = [
        tgt.named_columns[f"target_block{i}"]
        for i in range(patchnpacker.num_prediction_targets)
    ]
    tgt_patches = tgt_patches / 255
    tgt_attn_mask = get_attention_mask(tgt_image_ids)

    with torch.no_grad():
        with accelerator.autocast():
            tgt_states = teacher(tgt_patches, tgt_attn_mask, tgt_positions)
            tgt_ids = lfq(tgt_states, return_indices=True, return_dict=True)["indices"]

    import bpdb

    bpdb.set_trace()

    # Compute the context hidden states by using the vit with gradients enabled
    ctx_patches = ctx.named_columns["patches"]
    ctx_positions = ctx.named_columns["positions"]
    ctx_image_ids = ctx.named_columns["image_ids"]
    ctx_attn_mask = get_attention_mask(ctx_image_ids)

    # u8 -> float
    ctx_patches = ctx_patches / 255
    with accelerator.autocast():
        ctx_states = vit(ctx_patches, ctx_attn_mask, ctx_positions)

    tgt_batch_size, tgt_sequence_length = tgt.all_columns[0].shape[:2]
    tgt_pred_mask = torch.ones(
        tgt_batch_size, tgt_sequence_length, device=device, dtype=torch.bool
    )
    tgt = TensorSequence(
        named_columns={
            "states": tgt_states,
            "positions": tgt_positions,
            "image_ids": tgt_image_ids,
            "pred_mask": tgt_pred_mask,
        },
        sequence_dim=tgt.sequence_dim,
    )

    ctx_batch_size = ctx_patches.shape[0]
    ctx_sequence_length = ctx_patches.shape[1]

    # building block for masking out preds
    # pred_mask is 1 where the patch is a prediction instead of a context
    ctx_pred_mask = torch.zeros(
        ctx_batch_size, ctx_sequence_length, device=device, dtype=torch.bool
    )

    ctx = TensorSequence(
        named_columns={
            "states": ctx_states,
            "positions": ctx_positions,
            "image_ids": ctx_image_ids,
            "pred_mask": ctx_pred_mask,
        },
        sequence_dim=ctx.sequence_dim,
    )

    # Compute the loss from each target block mask and take the mean
    # For each block, the target states to predict are concatenated with the context states

    all_loss = []

    for tgt_block_mask in tgt_block_masks:
        tgt_preds = patchnpacker.make_prediction_target_sequence(
            tgt, ctx, tgt_block_mask
        )

        pred_states = tgt_preds.named_columns["states"]
        pred_positions = tgt_preds.named_columns["positions"]
        pred_image_ids = tgt_preds.named_columns["image_ids"]
        pred_tgt_mask = tgt_preds.named_columns["pred_mask"]
        # redo attention mask
        pred_attn_mask = get_attention_mask(pred_image_ids)

        with accelerator.autocast():
            y = predictor(pred_states, pred_attn_mask, pred_tgt_mask, pred_positions)

        # l1 loss

        # precision issues with masked mean here
        # loss = masked_mean(y - pred_states, pred_tgt_mask).mean()

        loss = F.l1_loss(y[pred_tgt_mask], pred_states[pred_tgt_mask])

        all_loss.append(loss)

    loss = torch.stack(all_loss).mean()
    return loss


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

    teacher = EMA(vit, **config.train.ema)
    print("vit: ", end="")
    print_num_parameters(vit)
    predictor = Predictor()
    print("predictor: ", end="")
    print_num_parameters(predictor)
    print("lfq: ", end="")
    print_num_parameters(lfq)

    if config.torch_compile:
        vit.forward = torch.compile(vit.forward)
        teacher.forward = torch.compile(teacher.forward)
        predictor.forward = torch.compile(predictor.forward)
        lfq.forward = torch.compile(lfq.forward)

    max_res = config.model.vit.patch_size * min(
        config.model.vit.max_height, config.model.vit.max_width
    )
    dataset = get_dataset(max_res=max_res, **config.train.dataset)

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
        vit, predictor, lfq, teacher, optimizer
    )

    id = 1

    step = 0

    dataloader = iter(dataloader)

    for ctx, tgt in patchnpacker.make_iter(dataloader):
        optimizer.zero_grad()

        ctx.to_device(device)
        tgt.to_device(device)

        loss = compute_training_loss(
            vit,
            lfq,
            predictor,
            accelerator,
            ctx,
            tgt,
            patchnpacker,
        )

        accelerator.backward(loss)
        optimizer.step()

        teacher.update()

        print(f"train loss: {loss.item():.5f} step {step}")
        wandb.log({"train": {"loss": loss}}, step=step)

        if (step + 1) % config.train.eval_every_num_steps == 0:
            loss = eval_classification_probe(
                vit, copy.deepcopy(predictor), config.eval, None, accelerator
            )
            wandb.log({"eval": {"loss": loss}})
            vit.train()
            predictor.train()

        step += 1


if __name__ == "__main__":
    main()
