from typing import Optional
import torch
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, OmegaConf
import accelerate
from torch.utils.data import DataLoader

from ijepa_enhanced.tensorset import TensorSet

from ..vit import ViT
from ..ema import EMA
from ..predictor import Predictor
from ..lfq import LFQ, masked_mean
from ..patchnpack import (
    MASK_IMAGE_ID,
    ContextTargetPatchNPacker,
    PatchNPacker,
    get_attention_mask,
)
from ..utils import print_num_parameters
from ..dataset import get_dataset


def train_step(vit, predictor, optimizer):
    pass


@hydra.main(version_base=None, config_path="../../conf", config_name="conf")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    vit = ViT(**config.vit)
    teacher = EMA(vit, **config.train.ema)
    print("vit: ", end="")
    print_num_parameters(vit)
    predictor = Predictor()
    print("predictor: ", end="")
    print_num_parameters(predictor)

    max_res = config.vit.patch_size * min(config.vit.max_height, config.vit.max_width)
    dataset = get_dataset(max_res=max_res, **config.train.data)

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=config.train.num_workers,
    )

    patchnpacker = ContextTargetPatchNPacker(
        patch_size=config.vit.patch_size,
        batch_size=config.train.batch_size,
        sequence_length_context=config.train.sequence_length_context,
        sequence_length_target=config.train.sequence_length_target,
    )

    optimizer_cls = getattr(torch.optim, config.train.optimizer.name)
    optimizer = optimizer_cls(
        list(vit.parameters()) + list(predictor.parameters()),
        **config.train.optimizer.args,
    )

    accelerator = accelerate.Accelerator()
    device = accelerator.device

    vit, predictor, teacher, optimizer = accelerator.prepare(
        vit, predictor, teacher, optimizer
    )

    training_done = False

    id = 1

    dataloader = iter(dataloader)

    while not training_done:
        if not patchnpacker.can_pop_batch():
            image = None
            try:
                image = next(dataloader)["pixel_values"]
            except StopIteration:
                training_done = True

            if image is not None:
                patchnpacker.append(image, id)
            id += 1
            continue

        optimizer.zero_grad()

        ctx, tgt, *pred_tgt = patchnpacker.pop_batch()

        tgt.to_device_(device)
        tgt_patches, tgt_positions, tgt_image_ids, *tgt_block_masks, tgt_attn_mask = (
            tgt.columns
        )
        tgt_patches = tgt_patches / 255

        tgt_batch_size = tgt_patches.shape[0]
        tgt_sequence_length = tgt_patches.shape[1]

        with torch.no_grad():
            with accelerator.autocast():
                tgt_states = teacher(
                    tgt_patches, tgt_attn_mask, *tgt_positions.unbind(-1)
                )

        ctx.to_device_(device)
        ctx_patches, ctx_positions, ctx_image_ids, ctx_attn_mask = ctx.columns
        ctx_patches = ctx_patches / 255

        ctx_batch_size = ctx_patches.shape[0]
        ctx_sequence_length = ctx_patches.shape[1]

        with accelerator.autocast():
            ctx_states = vit(ctx_patches, ctx_attn_mask, *ctx_positions.unbind(-1))

        # replaces the tgt patches with the tgt_states
        tgt = TensorSet([tgt_states, tgt_positions, tgt_image_ids], is_batched=True)

        # masked out preds
        ctx_pred_mask = torch.zeros(
            ctx_batch_size, ctx_sequence_length, device=device, dtype=torch.bool
        )

        ctx = TensorSet(
            [ctx_states, ctx_positions, ctx_image_ids, ctx_pred_mask], is_batched=True
        )

        all_loss = []

        for tgt_block_mask in tgt_block_masks:
            tgt_preds = []
            for seq_mask, tgt_seq in zip(tgt_block_mask, tgt):
                tgt_seq = tgt_seq[seq_mask]

                # creates mask
                pred_mask_seq = torch.ones(
                    tgt_seq.num_rows, device=device, dtype=torch.bool
                )

                # pads
                pad_amt = config.train.sequence_length_prediction - tgt_seq.num_rows
                assert (
                    pad_amt >= 0
                ), f"prediction sequence length too long by {-pad_amt}"
                if pad_amt > 0:
                    tgt_seq = tgt_seq.pad(pad_amt, MASK_IMAGE_ID)

                pred_mask_seq = torch.cat(
                    (
                        pred_mask_seq,
                        torch.zeros(pad_amt, device=device, dtype=torch.bool),
                    )
                )
                tgt_seq.columns.append(pred_mask_seq)

                tgt_preds.append(tgt_seq)

            tgt_preds = TensorSet.stack(tgt_preds)
            tgt_preds = TensorSet.cat([tgt_preds, ctx])

            pred_states, pred_positions, pred_image_ids, pred_tgt_mask = (
                tgt_preds.columns
            )
            # redo attention mask
            pred_attn_mask = get_attention_mask(pred_image_ids)

            with accelerator.autocast():
                y = predictor(
                    pred_states,
                    pred_attn_mask,
                    pred_tgt_mask,
                    *pred_positions.unbind(-1),
                )

            # l1
            # loss = masked_mean(y - pred_states, pred_tgt_mask).mean()
            loss = F.l1_loss(y[pred_tgt_mask], pred_states[pred_tgt_mask])
            all_loss.append(loss)

        loss = torch.cat(tuple(loss.unsqueeze(0) for loss in all_loss)).mean()
        accelerator.backward(loss)
        optimizer.step()

        print(loss.item())


if __name__ == "__main__":
    main()
