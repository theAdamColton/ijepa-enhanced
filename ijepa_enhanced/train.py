import torch.distributed as dist
import einx
import time
import omegaconf
from tqdm import tqdm
import torch.nn.functional as F
import math
import torch
import wandb

from ijepa_enhanced.mask import MakeContextPredMaskArgs

from .data import MASK_ID, get_dataloader
from .probe_eval import mean_by_id, probe_eval
from .io import save


class CosineAnnealingWarmup:
    def __init__(self, optim, max_steps, start_lr, lr, warmup_steps, start_step=0):
        self.optim = optim
        self.i = start_step

        def get_lr(step):
            if step <= warmup_steps:
                p = (step + 1) / warmup_steps
                return (lr - start_lr) * p + start_lr
            else:
                step -= warmup_steps
                return 0.5 * lr * (1 + math.cos(math.pi * step / max_steps))

        self.get_lr = get_lr

    @property
    def lr(self):
        return self.get_lr(self.i)

    def step(self):
        lr = self.get_lr(self.i)
        for group in self.optim.param_groups:
            group["lr"] = lr
        self.i += 1


def inplace_lerp(
    tgt: torch.Tensor,
    src: torch.Tensor,
    weight,
):
    if tgt.dtype != src.dtype:
        src = src.to(tgt.dtype)
    tgt.lerp_(src, weight)


def ema_update(ema_model, model, beta):
    for param_model, param_ema_model in zip(model.parameters(), ema_model.parameters()):
        if not param_model.is_floating_point():
            continue
        inplace_lerp(param_ema_model, param_model, 1 - beta)

    for buffer_model, buffer_ema_model in zip(model.buffers(), ema_model.buffers()):
        if not buffer_model.is_floating_point():
            continue
        inplace_lerp(buffer_ema_model, buffer_model, 1 - beta)
    return ema_model


def smooth_rank(x, eps=1e-7):
    """
    x: Batch of representations, shape: (B Z)

    This is a metric studied in the 2023 paper:
    RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by Their Rank

    Higher smooth rank
    """
    x = x[:10_000].float()

    s = torch.linalg.svdvals(x)
    s_norm = s.norm(1)
    p = s / s_norm
    log_p = torch.log(p + eps)
    entropy = torch.exp(-(p * log_p).sum()).item()
    return entropy


def compute_loss(
    patches,
    student,
    student_r,
    teacher,
    teacher_r,
    predictor,
    sequence_length_context,
    should_unmerge,
    should_log,
    merge_mode,
):
    log_dict = {}

    if should_log:
        log_dict["n_samples"] = sum(
            torch.unique(seq[seq != MASK_ID]).nelement()
            for seq in patches["sequence_ids"]
        )

        log_dict["percent_padding_all"] = (
            (patches["sequence_ids"] == MASK_ID).float().mean().item()
        )
        log_dict["percent_padding_pred"] = (
            (patches["sequence_ids"][:, sequence_length_context:] == MASK_ID)
            .float()
            .mean()
            .item()
        )
        log_dict["percent_padding_context"] = (
            (patches["sequence_ids"][:, :sequence_length_context] == MASK_ID)
            .float()
            .mean()
            .item()
        )

    prediction_block_masks = patches.named_columns.pop("prediction_block_masks")

    with torch.no_grad():
        teacher_outputs = teacher(
            patches["patches"],
            patches["height_ids"],
            patches["width_ids"],
            patches["sequence_ids"],
            r=teacher_r,
            mode=merge_mode,
        )
        teacher_hidden_states = teacher_outputs.hidden_states
        teacher_tome = teacher_outputs.tome

        if should_log:
            log_dict["smooth_rank"] = smooth_rank(
                mean_by_id(
                    teacher_hidden_states,
                    teacher_tome.merged_ids,
                ),
            )

        if should_unmerge:
            teacher_hidden_states = teacher_tome.unmerge_all(teacher_hidden_states)
            teacher_hidden_states = teacher_hidden_states[:, sequence_length_context:]

        teacher_hidden_states = F.layer_norm(
            teacher_hidden_states, (teacher_hidden_states.size(-1),)
        )

    context = patches.iloc[:, :sequence_length_context]

    student_outputs = student(
        context["patches"],
        context["height_ids"],
        context["width_ids"],
        context["sequence_ids"],
        r=student_r,
        mode=merge_mode,
    )

    student_hidden_states = student_outputs.hidden_states

    student_tome = student_outputs.tome

    if should_unmerge:
        student_hidden_states = student_tome.unmerge_all(student_hidden_states)

    # prepares student and teacher outputs for the predictor

    num_prediction_blocks = prediction_block_masks.size(-1)

    predictor_hidden_states = torch.cat(
        (student_hidden_states, teacher_hidden_states), 1
    )

    # repeats batch elements to match the number of prediction block masks
    predictor_hidden_states = einx.rearrange(
        "b ... -> (h b) ...", predictor_hidden_states, h=num_prediction_blocks
    )

    context_hidden_sequence_length = student_hidden_states.shape[1]

    if should_unmerge:
        sequence_ids = einx.rearrange(
            "b ... -> (h b) ...", patches["sequence_ids"], h=num_prediction_blocks
        )
        height_ids = einx.rearrange(
            "b ... -> (h b) ...", patches["height_ids"], h=num_prediction_blocks
        )
        width_ids = einx.rearrange(
            "b ... -> (h b) ...", patches["width_ids"], h=num_prediction_blocks
        )

        prediction_block_masks = einx.rearrange(
            "b s h -> (h b) s", prediction_block_masks
        )

        is_context = torch.zeros_like(prediction_block_masks)
        is_context[:, :context_hidden_sequence_length] = 1

        predictor_sequence_mask = ~(is_context | prediction_block_masks)

        predictor_hidden_states = predictor(
            predictor_hidden_states,
            height_ids,
            width_ids,
            prediction_block_masks,
            sequence_ids,
            sequence_mask=predictor_sequence_mask,
        ).hidden_states

    else:
        student_sequence_ids = student_outputs.tome.merged_ids
        teacher_sequence_ids = teacher_tome.merged_ids
        sequence_ids = torch.cat((student_sequence_ids, teacher_sequence_ids), 1)
        sequence_ids = einx.rearrange(
            "b ... -> (h b) ...", sequence_ids, h=num_prediction_blocks
        )

        prediction_block_masks = (
            teacher_tome.merge_all(prediction_block_masks * 1.0) > 0.0
        )
        prediction_block_masks = einx.rearrange(
            "b s h -> (h b) s", prediction_block_masks
        )
        prediction_block_masks = torch.cat(
            (
                torch.zeros(
                    prediction_block_masks.size(0),
                    context_hidden_sequence_length,
                    dtype=torch.bool,
                    device=prediction_block_masks.device,
                ),
                prediction_block_masks,
            ),
            1,
        )

        is_context = torch.zeros_like(prediction_block_masks)
        is_context[:, :context_hidden_sequence_length] = 1

        teacher_position_embeds = teacher_outputs.position_embeds
        teacher_position_embeds = teacher_tome.merge_all(teacher_position_embeds)

        student_position_embeds = student_outputs.position_embeds
        student_position_embeds = student_tome.merge_all(student_position_embeds)

        predictor_position_embeds = torch.cat(
            (teacher_position_embeds, student_position_embeds), 1
        )
        predictor_position_embeds = einx.rearrange(
            "b ... -> (h b) ...", predictor_position_embeds, h=num_prediction_blocks
        )

        predictor_sequence_mask = ~(is_context | prediction_block_masks)

        predictor_hidden_states = predictor(
            predictor_hidden_states,
            predictor_position_embeds,
            prediction_block_masks,
            sequence_ids,
            sequence_mask=predictor_sequence_mask,
        ).hidden_states

    # takes only the hidden states that represent possible target tokens
    predictor_hidden_states = predictor_hidden_states[
        :, context_hidden_sequence_length:
    ]
    prediction_block_masks = prediction_block_masks[:, context_hidden_sequence_length:]

    predictor_hidden_states = predictor_hidden_states[prediction_block_masks]
    teacher_hidden_states = einx.rearrange(
        "b ... -> (h b) ...", teacher_hidden_states, h=num_prediction_blocks
    )
    teacher_hidden_states = teacher_hidden_states[prediction_block_masks]

    loss = F.smooth_l1_loss(predictor_hidden_states, teacher_hidden_states)

    log_dict["loss"] = loss.item()
    return loss, log_dict


class BetaScheduler:
    def __init__(self, start_beta=0.996, steps=1_000_000, end_beta=1.0, start_step=0):
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.steps = steps
        self.i = start_step

    @property
    def beta(self):
        return (self.end_beta - self.start_beta) * (
            self.i / self.steps
        ) + self.start_beta

    def step(self):
        self.i += 1


def get_is_main_proc():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0 or dist.get_world_size() == 1
    return True


def train(conf, optimizer, scaler, teacher, student, predictor, global_step, epoch):
    device = "cuda"

    is_main_proc = get_is_main_proc()
    is_dist_training = dist.is_initialized()

    should_torch_compile = conf.torch_compile

    mask_args = MakeContextPredMaskArgs(**conf.train.mask_args)
    dataloader = get_dataloader(
        batch_size=conf.train.batch_size,
        make_context_pred_mask_args=mask_args,
        **conf.dataset_train,
    )

    sequence_length_context = conf.dataset_train.sequence_length_context

    num_train_steps = conf.train.num_steps
    scheduler_scale = conf.train.scheduler_scale
    num_warmup_steps = conf.train.warmup_steps
    run_eval_every_num_steps = conf.train.run_eval_every_num_steps
    save_every_num_steps = conf.train.save_every_num_steps
    lr = conf.train.lr

    scheduler = CosineAnnealingWarmup(
        optim=optimizer,
        max_steps=num_train_steps * scheduler_scale,
        start_lr=2e-4,
        lr=lr,
        warmup_steps=num_warmup_steps,
        start_step=global_step,
    )

    beta_scheduler = BetaScheduler(
        conf.train.start_beta,
        steps=num_train_steps * scheduler_scale,
        start_step=global_step,
    )

    run = wandb.init(
        project="ijepa-enhanced-v3",
        config=omegaconf.OmegaConf.to_container(conf, resolve=True),
    )

    run_name = run.name

    compute_loss_fn = compute_loss
    if should_torch_compile:
        compute_loss_fn = torch.compile(compute_loss_fn)

    student_r = conf.train.student_r
    teacher_r = conf.train.teacher_r
    merge_mode = conf.train.merge_mode
    should_unmerge = conf.train.should_unmerge

    prog_bar = tqdm(
        total=conf.train.num_steps, initial=global_step, disable=not is_main_proc
    )
    prev_time = time.time()
    while True:
        for batch in dataloader:
            run_eval = (global_step + 1) % run_eval_every_num_steps == 0
            should_log = (global_step % conf.train.log_every_num_steps == 0) or run_eval

            log_dict = {}

            patches, _ = batch
            patches = patches.to_device("cuda")

            with torch.autocast(device, torch.bfloat16):
                loss, train_log_dict = compute_loss(
                    patches=patches,
                    student=student,
                    student_r=student_r,
                    teacher=teacher,
                    teacher_r=teacher_r,
                    predictor=predictor,
                    sequence_length_context=sequence_length_context,
                    should_unmerge=should_unmerge,
                    should_log=should_log,
                    merge_mode=merge_mode,
                )

            log_dict["train"] = train_log_dict

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            beta_scheduler.step()
            scheduler.step()

            with torch.no_grad():
                ema_update(teacher, student, beta=beta_scheduler.beta)

            step_duration = time.time() - prev_time

            log_dict["train"]["step_duration"] = step_duration
            log_dict["train"]["epoch"] = epoch
            log_dict["train"]["lr"] = scheduler.lr
            log_dict["train"]["beta"] = beta_scheduler.beta

            desc = "".join(f"{k}:{v}" for k, v in log_dict.items())

            prog_bar.set_description(desc)
            prog_bar.update(1)

            if run_eval:
                if is_main_proc:
                    del loss
                    probe_accuracy, probe_train_val_accuracy = probe_eval(conf, teacher)

                    log_dict["eval"] = {}
                    log_dict["eval"]["probe_accuracy"] = probe_accuracy
                    log_dict["eval"][
                        "probe_accuracy_train_val"
                    ] = probe_train_val_accuracy

                if is_dist_training:
                    dist.barrier()

            if (global_step + 1) % save_every_num_steps == 0 and is_main_proc:
                save(
                    run_name,
                    global_step,
                    epoch,
                    should_torch_compile,
                    teacher,
                    student,
                    predictor,
                    optimizer,
                    scaler,
                )

            if should_log and is_main_proc:
                wandb.log(log_dict, step=global_step)

            prev_time = time.time()
            global_step += 1
        epoch += 1

        if global_step >= num_train_steps:
            break

    prog_bar.close()

    if is_main_proc:
        save(
            run_name,
            global_step,
            epoch,
            should_torch_compile,
            teacher,
            student,
            predictor,
            optimizer,
            scaler,
        )
