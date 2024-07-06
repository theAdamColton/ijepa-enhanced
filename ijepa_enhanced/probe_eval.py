import math
from tqdm import tqdm
import torch.nn.functional as F
from torch import nn
import torch

from .data import get_dataloader, MASK_ID
from .ops import mean_by_id


def collect_labels(metadata_batch, sequence_ids):
    per_image_labels = []
    for metadata, ids in zip(metadata_batch, sequence_ids):
        ids = torch.unique(ids)
        for id in ids:
            if id == MASK_ID:
                continue
            label = metadata[id.item()]["label"]
            per_image_labels.append(label)
    per_image_labels = torch.tensor(per_image_labels, dtype=torch.long, device="cpu")
    return per_image_labels


def get_probes_optim_scheduler(lrs, device, t_max, input_size, output_size):
    optim_param_groups = []
    probes = []
    for lr in lrs:
        probe = nn.Linear(input_size, output_size).to(device)
        probe.weight.data.normal_(0, 0.01)
        probe.bias.data.zero_()
        probes.append(probe)
        optim_param_groups.append({"params": probe.parameters(), "lr": lr})
    optim = torch.optim.SGD(optim_param_groups, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, t_max)

    return probes, optim, scheduler


def scale_lr(learning_rates, batch_size):
    return learning_rates * batch_size / 256.0


def probe_eval(conf, encoder):
    tome_r = conf.probe_eval.r
    tome_enabled = tome_r > 0
    should_unmerge = conf.probe_eval.should_unmerge

    encoder_dtype = torch.bfloat16
    device = "cuda"

    def run_batch(batch):
        patches, metadata = batch
        patches = patches.to_device(device)
        sequence_ids = patches.named_columns.pop("sequence_ids")
        with torch.inference_mode():
            with torch.autocast(device, encoder_dtype):
                outputs = encoder(
                    patches["patches"],
                    patches["height_ids"],
                    patches["width_ids"],
                    sequence_ids,
                    r=tome_r,
                )

                hidden_states = outputs.hidden_states

                if should_unmerge and tome_enabled:
                    hidden_states = outputs.tome.unmerge_all(hidden_states)
                if tome_enabled and not should_unmerge:
                    sequence_ids = outputs.tome.merged_ids

            hidden_states = hidden_states.float()
            per_image_hidden_states = mean_by_id(hidden_states, sequence_ids)
            per_image_hidden_states = F.layer_norm(
                per_image_hidden_states, (per_image_hidden_states.size(-1),)
            )
        labels = collect_labels(metadata, sequence_ids)
        return per_image_hidden_states, labels

    dataset_conf = conf.dataset_train_probe
    num_samples = dataset_conf.num_samples
    batch_size = conf.probe_eval.batch_size_encoder

    dataloader = get_dataloader(
        batch_size=batch_size,
        make_context_pred_mask_args=None,
        do_shuffle=False,
        **dataset_conf,
    )

    all_embeddings = []
    all_labels = []
    prog_bar = tqdm(total=num_samples, desc="embedding...")
    for batch in dataloader:
        per_image_hidden_states, labels = run_batch(batch)
        all_embeddings.append(per_image_hidden_states.to("cpu"))
        all_labels.append(labels)

        if sum(len(x) for x in all_labels) > num_samples:
            print("stopping embedding...")
            break

        prog_bar.update(per_image_hidden_states.size(0))
    prog_bar.close()

    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels)

    val_split = conf.probe_eval.val_split

    val_split_size = int(len(all_labels) * val_split)

    val_embeddings, all_embeddings = (
        all_embeddings[:val_split_size],
        all_embeddings[val_split_size:],
    )
    val_labels, all_labels = all_labels[:val_split_size], all_labels[val_split_size:]

    batch_size_probe = conf.probe_eval.batch_size_probe
    num_epochs = conf.probe_eval.num_epochs
    num_probe_steps = (
        int(math.ceil(len(all_embeddings) / batch_size_probe)) * num_epochs
    )
    scheduler_scale = conf.probe_eval.scheduler_scale

    lrs = [5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8]
    scaled_lrs = [scale_lr(lr, batch_size) for lr in lrs]
    probes, optimizer, scheduler = get_probes_optim_scheduler(
        scaled_lrs, device, num_probe_steps * scheduler_scale, encoder.hidden_size, 1000
    )

    def compute_val_acc(probe):
        with torch.inference_mode():
            logits = probe(val_embeddings.to(device))
            preds = logits.argmax(-1)
            acc = (preds == val_labels.to(device)).float().mean().item()
        return acc

    for epoch in tqdm(range(num_epochs)):
        rand_perm = torch.randperm(len(all_embeddings))
        all_embeddings = all_embeddings[rand_perm]
        all_labels = all_labels[rand_perm]

        all_probe_train_accs = [[] for _ in range(len(probes))]
        all_probe_train_losses = [[] for _ in range(len(probes))]
        for embeddings, labels in zip(
            all_embeddings.split(batch_size_probe, 0),
            all_labels.split(batch_size_probe, 0),
        ):
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            for probe, lr, all_train_accs, all_train_losses in zip(
                probes, lrs, all_probe_train_accs, all_probe_train_losses
            ):
                logits = probe(embeddings)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                preds = logits.argmax(-1)
                train_acc = (preds == labels).float().mean().item()
                all_train_accs.append(train_acc)
                all_train_losses.append(loss.item())
            optimizer.step()
            scheduler.step()

        val_accs = [compute_val_acc(probe) for probe in probes]
        best_val_lr_acc = (None, -1, -1)
        for lr, all_train_accs in zip(lrs, val_accs):
            acc = torch.tensor(all_train_accs).mean().item()
            if acc > best_val_lr_acc[1]:
                best_val_lr_acc = (lr, acc)

        best_lr_acc_loss = (None, -1, -1)
        for lr, all_train_accs, all_train_losses in zip(
            lrs, all_probe_train_accs, all_probe_train_losses
        ):
            acc = torch.tensor(all_train_accs).mean().item()
            loss = torch.tensor(all_train_losses).mean().item()
            if acc > best_lr_acc_loss[1]:
                best_lr_acc_loss = (lr, acc, loss)

        print(
            epoch,
            "best train lr,acc,loss:",
            best_lr_acc_loss,
            "best val lr,acc:",
            best_val_lr_acc,
            "lr",
            scheduler.get_last_lr()[0],
        )

    dataloader = get_dataloader(
        batch_size=conf.probe_eval.batch_size_encoder,
        make_context_pred_mask_args=None,
        do_shuffle=False,
        **conf.dataset_val,
    )

    del optimizer

    all_probe_preds = [[] for _ in range(len(probes))]
    all_probe_labels = [[] for _ in range(len(probes))]
    num_val_samples = conf.dataset_val.num_samples
    prog_bar = tqdm(total=num_val_samples, desc="running eval predictions")
    for batch in dataloader:
        per_image_hidden_states, labels = run_batch(batch)
        for probe, all_preds, all_labels in zip(
            probes, all_probe_preds, all_probe_labels
        ):
            with torch.inference_mode():
                logits = probe(per_image_hidden_states)
                preds = logits.argmax(-1)
            all_preds.append(preds.to("cpu"))
            all_labels.append(labels)

        prog_bar.update(len(labels))
    prog_bar.close()

    best_acc = -1
    for lr, all_preds, all_labels in zip(lrs, all_probe_preds, all_probe_labels):
        acc = (torch.cat(all_preds) == torch.cat(all_labels)).float().mean().item()
        print(f"lr: {lr} val acc", acc)
        if acc > best_acc:
            best_acc = acc

    return best_acc, best_val_lr_acc[-1]
