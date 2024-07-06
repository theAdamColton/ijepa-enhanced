import os
import torch


def save(
    run_name,
    global_step,
    epoch,
    should_torch_compile,
    teacher,
    student,
    predictor,
    optimizer,
    scaler,
):
    save_path = f"out/{run_name}/{global_step:06}.pt"
    print("saving", save_path)
    os.makedirs("out/", exist_ok=True)
    os.makedirs(f"out/{run_name}/", exist_ok=True)
    torch.save(
        dict(
            teacher=(
                teacher._orig_mod.state_dict() if should_torch_compile else teacher
            ),
            student=(
                student._orig_mod.state_dict() if should_torch_compile else student
            ),
            predictor=(
                predictor._orig_mod.state_dict() if should_torch_compile else predictor
            ),
            optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),
            global_step=global_step,
            epoch=epoch,
        ),
        save_path,
    )


def load(optimizer, scaler, teacher, student, predictor, path):
    state_dict = torch.load(path, map_location="cpu")
    global_step = state_dict["global_step"]
    epoch = state_dict["epoch"]

    device = teacher.patch_embed.weight.device

    def _to(d):
        return {
            k: (v.to(device=device) if isinstance(v, torch.Tensor) else v)
            for k, v in d.items()
        }

    optimizer.load_state_dict(_to(state_dict.pop("optimizer")))
    scaler.load_state_dict(_to(state_dict.pop("scaler")))
    teacher.load_state_dict(_to(state_dict.pop("teacher")))
    student.load_state_dict(_to(state_dict.pop("student")))
    predictor.load_state_dict(_to(state_dict.pop("predictor")))

    print("loaded state", path)

    return global_step, epoch
