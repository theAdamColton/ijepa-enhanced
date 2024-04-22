import einx
from tqdm import tqdm
import torch
import accelerate
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .dataset import get_dataset
from .patchnpack import MASK_IMAGE_ID, PatchNPacker, get_attention_mask
from .optimizer import get_optimizer


def make_pred(ctx, vit, lfq, predictor, predictor_head, accelerator):
    ctx_patches = ctx["patches"]
    ctx_positions = ctx["positions"]
    ctx_image_ids = ctx["image_ids"]
    ctx_labels = ctx["label"]

    ctx_attn_mask = get_attention_mask(ctx_image_ids)
    ctx_patches = ctx_patches / 255

    with torch.no_grad():
        with accelerator.autocast():
            hidden_states = vit(ctx_patches, ctx_attn_mask, ctx_positions)
            hidden_states = lfq(hidden_states, return_dict=True)["hidden_states"]

    with accelerator.autocast():
        tgt_mask = torch.zeros_like(ctx_attn_mask[:, 0])
        tgt_hidden_states = predictor(
            hidden_states, ctx_attn_mask, tgt_mask, ctx_positions
        )

    # combines the features at each unique id by taking the mean
    all_hidden_states = []
    all_labels = []
    for id in torch.unique(ctx_image_ids):
        if id == MASK_IMAGE_ID:
            continue
        mask = ctx_image_ids == id
        hidden_states = tgt_hidden_states[mask].mean(0)

        label = ctx_labels[mask][0]
        all_labels.append(label)

        all_hidden_states.append(hidden_states)

    all_hidden_states = torch.stack(all_hidden_states)

    all_hidden_states = einx.rearrange("b h z -> b (h z)", all_hidden_states)

    with accelerator.autocast():
        logits = predictor_head(all_hidden_states)

    all_labels = torch.stack(all_labels)
    loss = F.cross_entropy(logits, all_labels)

    return loss, logits, all_labels


def eval_classification_probe(
    vit,
    lfq,
    predictor,
    config,
    predictor_head=None,
    accelerator=None,
):
    if accelerator is None:
        accelerator = accelerate.Accelerator()

    device = accelerator.device

    vit = vit.eval()
    predictor = predictor.train()
    if predictor_head is None:
        predictor_head = torch.nn.Linear(
            predictor.projection_dim * predictor.projection_heads,
            config.dataset.num_classes,
            bias=False,
            device=device,
        )

    if config.torch_compile:
        vit.forward = torch.compile(vit.forward)
        predictor.forward = torch.compile(predictor.forward)
        predictor_head.forward = torch.compile(predictor_head.forward)

    patchnpacker = PatchNPacker(
        vit.patch_size, config.sequence_length, config.batch_size
    )
    optimizer = get_optimizer(
        config.optimizer,
        list(predictor.parameters()) + list(predictor_head.parameters()),
    )

    train_dataset = get_dataset(split="train", **config.dataset)
    dataloader = DataLoader(
        train_dataset,
        num_workers=config.num_workers,
        collate_fn=None,
        batch_size=None,
    )
    dataloader = iter(dataloader)

    vit, predictor, lfq, predictor_head, optimizer = accelerator.prepare(
        vit, predictor, lfq, predictor_head, optimizer
    )

    step = 0
    _id = 1

    for ctx in patchnpacker.make_iter(dataloader):
        ctx.to_device(accelerator.device)

        loss, logits, labels = make_pred(
            ctx, vit, lfq, predictor, predictor_head, accelerator
        )

        print(f"eval loss {loss:.4f} step {step}")

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        step += 1

        if step > config.max_iterations:
            break

    patchnpacker = PatchNPacker(
        vit.patch_size, config.sequence_length, config.batch_size_validation
    )

    validation_dataset = get_dataset(split="validation", **config.dataset)
    dataloader = DataLoader(
        validation_dataset,
        num_workers=config.num_workers,
        collate_fn=None,
        batch_size=None,
    )
    dataloader = iter(dataloader)

    all_preds = []
    all_labels = []

    progress_bar = tqdm(desc="validation", total=len(validation_dataset))

    for ctx in patchnpacker.make_iter(dataloader):
        ctx.to_device(device)

        with torch.inference_mode():
            loss, logits, labels = make_pred(
                ctx, vit, lfq, predictor, predictor_head, accelerator
            )

        preds = logits.argmax(-1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

        progress_bar.update(len(preds))

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    accuracy = ((all_preds == all_labels) * 1.0).mean()

    print("accuracy:", accuracy)
