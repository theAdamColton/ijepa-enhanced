import einx
from tqdm import tqdm
import torch
import accelerate
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .dataset import get_dataset
from .patchnpack import MASK_IMAGE_ID, PatchNPacker, get_attention_mask
from .optimizer import get_optimizer


def make_pred(ctx, teacher, predictor, predictor_head, accelerator):
    ctx_patches = ctx["patches"]
    ctx_positions = ctx["positions"]
    ctx_image_ids = ctx["image_ids"]
    ctx_labels = ctx["label"]

    ctx_attn_mask = get_attention_mask(ctx_image_ids)
    ctx_patches = ctx_patches / 255

    with torch.no_grad():
        with accelerator.autocast():
            _, hidden_states = teacher(ctx_patches, ctx_attn_mask, ctx_positions)

    tgt_mask = torch.zeros_like(ctx_attn_mask[:, 0])
    with accelerator.autocast():
        hidden_states = predictor(hidden_states, ctx_attn_mask, tgt_mask, ctx_positions)

    # combines the features at each unique id by taking the mean
    all_hidden_states = []
    all_labels = []
    for id in torch.unique(ctx_image_ids):
        if id == MASK_IMAGE_ID:
            continue
        mask = ctx_image_ids == id
        image_hidden_states = hidden_states[mask].mean(0)
        label = ctx_labels[mask][0]
        all_labels.append(label)
        all_hidden_states.append(image_hidden_states)

    all_hidden_states = torch.stack(all_hidden_states)

    all_hidden_states = einx.rearrange("b h z -> b (h z)", all_hidden_states)

    with accelerator.autocast():
        logits = predictor_head(all_hidden_states)

    all_labels = torch.stack(all_labels)
    loss = F.cross_entropy(logits, all_labels)

    return loss, logits, all_labels


def eval_classification_probe(
    teacher,
    predictor,
    config,
    accelerator: accelerate.Accelerator,
    patch_size=None,
):
    """
    teacher: already prepared by accelerator
    predictor: already prepared by accelerator
    """
    device = accelerator.device

    predictor_head = torch.nn.Linear(
        predictor.projection_dim * predictor.projection_heads,
        config.eval.dataset.num_classes,
        bias=False,
        device=device,
    )

    patchnpacker = PatchNPacker(
        patch_size, config.eval.sequence_length, config.eval.batch_size
    )

    optimizer = get_optimizer(
        config.eval.optimizer,
        list(predictor.parameters()) + list(predictor_head.parameters()),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config.eval.max_iterations, 1e-7
    )

    train_dataset = get_dataset(**config.train.dataset)
    dataloader = DataLoader(
        train_dataset,
        num_workers=config.num_workers,
        collate_fn=None,
        batch_size=None,
    )

    predictor_head, optimizer, scheduler, dataloader = accelerator.prepare(
        predictor_head, optimizer, scheduler, dataloader
    )

    step = 0

    for ctx in patchnpacker.make_iter(dataloader):
        ctx.to_device(accelerator.device)

        loss, logits, labels = make_pred(
            ctx, teacher, predictor, predictor_head, accelerator
        )

        accelerator.print(f"eval loss {loss:.4f} step {step}")

        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        step += 1

        if step > config.eval.max_iterations:
            break

    patchnpacker = PatchNPacker(
        patch_size,
        config.eval.sequence_length,
        config.eval.batch_size_validation,
    )

    validation_dataset = get_dataset(**config.eval.dataset)
    dataloader = DataLoader(
        validation_dataset,
        num_workers=config.num_workers,
        collate_fn=None,
        batch_size=None,
    )

    dataloader = accelerator.prepare(dataloader)

    all_preds = []
    all_labels = []

    try:
        n = len(validation_dataset)
    except:
        n = None

    progress_bar = tqdm(
        desc="validation", total=n, disable=not accelerator.is_local_main_process
    )

    PAD_ID = -200

    for ctx in patchnpacker.make_iter(dataloader):
        ctx.to_device(device)

        with torch.inference_mode():
            loss, logits, labels = make_pred(
                ctx, teacher, predictor, predictor_head, accelerator
            )

        preds = logits.argmax(-1)

        preds, labels = accelerator.pad_across_processes((preds, labels), 0, PAD_ID)
        preds, labels = accelerator.gather_for_metrics((preds, labels))

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

        progress_bar.update(len(preds))
        if progress_bar.n > n:
            break

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_preds = all_preds[all_preds != PAD_ID]
    all_labels = all_labels[all_labels != PAD_ID]
    accuracy = ((all_preds == all_labels) * 1.0).mean().item()

    accelerator.print(f"accuracy:{accuracy}")

    return accuracy
