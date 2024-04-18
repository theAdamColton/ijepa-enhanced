import torch
import accelerate
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .dataset import get_dataset
from .patchnpack import MASK_IMAGE_ID, PatchNPacker
from .optimizer import get_optimizer


def eval_classification_probe(
    vit,
    predictor,
    config,
    predictor_head=None,
    accelerator=None,
    max_iterations=9000,
):

    patchnpacker = PatchNPacker(
        vit.patch_size, config.sequence_length, config.batch_size
    )
    optimizer = get_optimizer(config.optimizer, predictor.parameters())

    train_dataset = get_dataset(split="train", **config.dataset)
    dataloader = DataLoader(
        train_dataset, num_workers=config.num_workers, collate_fn=None, batch_size=None
    )
    dataloader = iter(dataloader)

    if accelerator is None:
        accelerator = accelerate.Accelerator()

    if predictor_head is None:
        predictor_head = torch.nn.Linear(
            predictor.hidden_size,
            config.dataset.num_classes,
            bias=False,
            device=accelerator.device,
        )

    vit, predictor, predictor_head, optimizer = accelerator.prepare(
        vit, predictor, predictor_head, optimizer
    )

    _id = 1

    id_to_label = dict()
    while True:
        if not patchnpacker.can_pop_batch():
            try:
                row = next(dataloader)
            except StopIteration:
                break
            pixel_values = row["pixel_values"]
            label = row["label"]
            id_to_label[_id] = label
            patchnpacker.append_image(pixel_values, _id)
            _id += 1
            continue

        ctx = patchnpacker.pop_batch()
        ctx.to_device_(accelerator.device)

        ctx_patches, ctx_positions, ctx_image_ids, ctx_attn_mask = ctx.columns
        ctx_patches = ctx_patches / 255

        with torch.no_grad():
            with accelerator.autocast():
                hidden_states = vit(ctx_patches, ctx_attn_mask, ctx_positions)

        with accelerator.autocast():
            tgt_mask = torch.zeros_like(ctx_attn_mask[:, 0])
            tgt_hidden_states = predictor(
                hidden_states, ctx_attn_mask, tgt_mask, ctx_positions
            )

        # combines the features at each unique id
        all_hidden_states = []
        all_labels = []
        for id in torch.unique(ctx_image_ids):
            if id == MASK_IMAGE_ID:
                continue
            mask = ctx_image_ids == id
            hidden_states = tgt_hidden_states[mask].mean(0)

            label = id_to_label.pop(int(id.item()))
            label = torch.LongTensor([label]).to(accelerator.device)

            all_hidden_states.append(hidden_states)
            all_labels.append(label)

        all_hidden_states = torch.stack(all_hidden_states)
        all_labels = torch.cat(all_labels)

        with accelerator.autocast():
            logits = predictor_head(all_hidden_states)

        loss = F.cross_entropy(logits, all_labels)
        print("classification loss", loss)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
