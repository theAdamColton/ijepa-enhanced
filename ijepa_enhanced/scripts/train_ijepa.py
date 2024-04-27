import gc
import safetensors.torch
import copy
import torch
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, OmegaConf
import accelerate
from torch.utils.data import DataLoader
import wandb

from tensorsequence import TensorSequence

from ..vit import ViT
from ..teacher import Teacher
from ..predictor import Predictor
from ..lfq import calculate_perplexity, LFQ
from ..patchnpack import (
    MASK_IMAGE_ID,
    ContextTargetPatchNPacker,
    get_attention_mask,
)
from ..eval import eval_classification_probe
from ..utils import print_num_parameters
from ..dataset import get_dataset
from ..optimizer import get_optimizer

import logging

log = logging.getLogger(__name__)


def compute_target_states(teacher: Teacher, tgt: TensorSequence, accelerator):
    # u8 to float
    tgt_patches = tgt["patches"] / 255
    tgt_image_ids = tgt["image_ids"]
    tgt_positions = tgt["positions"]
    tgt_attn_mask = get_attention_mask(tgt_image_ids)

    with torch.inference_mode():
        with accelerator.autocast():
            tgt_ids, tgt_states = teacher(tgt_patches, tgt_attn_mask, tgt_positions)

    return TensorSequence(
        named_columns={
            "ids": tgt_ids,
            "states": tgt_states,
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
    teacher: Teacher,
    predictor: Predictor,
    accelerator,
    ctx: TensorSequence,
    tgt: TensorSequence,
    patchnpacker: ContextTargetPatchNPacker,
):
    device = accelerator.device
    # Compute target hidden states by passing the target patches through the teacher network

    tgt = tgt.to_device(device)
    prediction_block_masks = tgt["prediction_block_masks"]
    tgt = compute_target_states(teacher, tgt, accelerator)

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

    with torch.inference_mode():
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
    torch.set_float32_matmul_precision("medium")

    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    wandb.init(
        name="ijepa-enhanced",
        config=OmegaConf.to_container(config, resolve=True),
        mode=config.wandb_mode,
    )

    vit = ViT(**config.model.vit)
    # safetensors.torch.load_model(
    #     vit, "./pretrained-models/vit-base-patch16.safetensors"
    # )
    lfq = LFQ(**config.model.lfq)
    predictor = Predictor(**config.model.predictor)
    # safetensors.torch.load_model(
    #     predictor, "./pretrained-models/predictor-base.safetensors"
    # )

    teacher = Teacher(vit, lfq, **config.train.ema)
    print("vit: ", end="")
    print_num_parameters(vit)
    print("predictor: ", end="")
    print_num_parameters(predictor)
    print("lfq: ", end="")
    print_num_parameters(lfq)

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

    project_configuration = accelerate.accelerator.ProjectConfiguration(
        project_dir=hydra_output_dir, **config.train.accelerator_project_configuration
    )
    accelerator = accelerate.Accelerator(
        device_placement=False,
        project_config=project_configuration,
        **config.train.accelerator_args,
    )

    device = accelerator.device

    vit = vit.to(device)
    predictor = predictor.to(device)
    teacher = teacher.to(device)
    lfq = lfq.to(device)

    vit, predictor, teacher, lfq, optimizer = accelerator.prepare(
        vit, predictor, teacher, lfq, optimizer
    )

    if config.train.accelerator_resume_path:
        print("loading state from ", config.train.accelerator_resume_path)
        accelerator.load_state(config.train.accelerator_resume_path)

    step = 0

    dataloader = iter(dataloader)

    accuracy = -9999999

    for ctx, tgt in patchnpacker.make_iter(dataloader):
        with accelerator.accumulate(vit, lfq, predictor):
            optimizer.zero_grad()

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
            del ctx, tgt

            loss = (
                config.train.commit_loss_weight * loss_dict["commit_loss"]
                + config.train.entropy_loss_weight * loss_dict["entropy_loss"]
                + loss_dict["prediction_loss"]
            )

            accelerator.backward(loss)
            optimizer.step()

            teacher.update()

            loss_stmt = " ".join([f"{k}:{v.item():.5f}" for k, v in loss_dict.items()])

            log.info(f"train loss: {loss.item():.5f} {loss_stmt} step {step}")
            wandb.log({"train": loss_dict}, step=step)

            del loss_dict

            if (step + 1) % config.train.eval_every_num_steps == 0:
                accuracy = eval_classification_probe(
                    teacher,
                    copy.deepcopy(predictor),
                    config,
                    None,
                    accelerator,
                    patch_size=vit.patch_size,
                )
                wandb.log({"eval": {"accuracy": accuracy}}, step=step)
                vit.train()
                predictor.train()

            if (step + 1) % config.train.save_every_num_steps == 0:
                accelerator.save_state(hydra_output_dir + "/checkpoint/")

            if step > config.train.max_steps:
                break

            step += 1

            gc.collect()
            torch.cuda.empty_cache()

    return -accuracy


if __name__ == "__main__":
    main()
