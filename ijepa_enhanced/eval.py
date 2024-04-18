import wandb
import torch
import accelerate
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .dataset import get_dataset
from .patchnpack import MASK_IMAGE_ID, PatchNPacker, get_attention_mask
from .optimizer import get_optimizer


def eval_classification_probe(
    vit,
    predictor,
    config,
    predictor_head=None,
    accelerator=None,
):
    vit = vit.eval()
    predictor = predictor.train()
    if predictor_head is None:
        predictor_head = torch.nn.Linear(
            predictor.hidden_size,
            config.dataset.num_classes,
            bias=False,
            device=accelerator.device,
        )

    patchnpacker = PatchNPacker(
        vit.patch_size, config.sequence_length, config.batch_size
    )
    optimizer = get_optimizer(
        config.optimizer,
        list(predictor.parameters()) + list(predictor_head.parameters()),
    )

    train_dataset = get_dataset(split="train", **config.dataset)
    dataloader = DataLoader(
        train_dataset, num_workers=config.num_workers, collate_fn=None, batch_size=None
    )
    dataloader = iter(dataloader)

    if accelerator is None:
        accelerator = accelerate.Accelerator()

    vit, predictor, predictor_head, optimizer = accelerator.prepare(
        vit, predictor, predictor_head, optimizer
    )

    step = 0
    _id = 1

    id_to_label = dict()
    for ctx in patchnpacker.make_iter(dataloader):
        ctx.to_device(accelerator.device)

        ctx_patches, ctx_positions, ctx_image_ids = ctx.columns
        ctx_attn_mask = get_attention_mask(ctx_image_ids)
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
        print(f"eval loss {loss:.4f} step {step}")

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        step += 1

        if step > config.max_iterations:
            break

    patchnpacker.reset()

    validation_dataset = get_dataset(split="validation", **config.dataset)
    dataloader = DataLoader(
        validation_dataset,
        num_workers=config.num_workers,
        collate_fn=None,
        batch_size=None,
    )
    dataloader = iter(dataloader)

    for ctx in patchnpacker.make_iter(dataloader):
        ctx.to_device(device)
        import bpdb

        bpdb.set_trace()
