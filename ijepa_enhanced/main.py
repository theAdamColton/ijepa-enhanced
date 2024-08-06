from torch import distributed as dist
import multiprocessing as mp
import torch
from einx import Optional
import omegaconf
import jsonargparse
from torch.nn.parallel import DistributedDataParallel

from .train import train
from .viz import viz
from . import io
from .vit import ViT, Predictor, PredictorPositionless
from .probe_eval import probe_eval
from . import vit_factory


def num_params(m):
    return sum(m.nelement() for m in m.parameters() if m.requires_grad)


def proc_main(
    conf_file: str = "conf/vit-s-patch14.yaml",
    entry: str = "train",
    resume_path: Optional[str] = None,
    devices: Optional[list[str]] = None,
    rank: int = 0,
):
    with open(conf_file) as f:
        conf = omegaconf.OmegaConf.load(f)

    scaler = torch.cuda.amp.GradScaler()

    student = ViT(**conf.vit)
    if conf.train.merge_pretrained_vit:
        vit_factory.merge(student)

    device = "cuda"

    teacher = ViT(**conf.vit)
    teacher.load_state_dict(student.state_dict())
    teacher = teacher.to(device)
    student = student.to(device)
    if conf.train.should_unmerge:
        predictor = Predictor(**conf.predictor).to(device)
    else:
        predictor = PredictorPositionless(**conf.predictor).to(device)

    if devices is not None:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", world_size=len(devices), rank=rank)

        student = DistributedDataParallel(student, static_graph=True)
        teacher = DistributedDataParallel(teacher, static_graph=True)
        predictor = DistributedDataParallel(predictor, static_graph=True)

    print("predictor num parameters", num_params(predictor))
    print("encoder num parameters", num_params(student))

    lr = conf.train.lr
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(predictor.parameters()),
        lr=lr,
        weight_decay=5e-2,
    )

    if resume_path is not None:
        global_step, epoch = io.load(
            optimizer, scaler, teacher, student, predictor, resume_path
        )
    else:
        global_step, epoch = 0, 0

    should_torch_compile = conf.torch_compile
    if should_torch_compile:
        torch._dynamo.config.cache_size_limit = 256
        student = torch.compile(student)
        teacher = torch.compile(teacher)
        predictor = torch.compile(predictor)

    if entry == "train":
        train(conf, optimizer, scaler, teacher, student, predictor, global_step, epoch)
    elif entry == "probe_eval":
        assert devices is None
        probe_eval(conf, teacher)
    elif entry == "viz":
        assert devices is None
        viz(conf)
    else:
        print("unrecognized entry point ", entry)

    if devices is not None:
        dist.destroy_process_group()


def main(
    conf_file: str = "conf/vit-s-patch14.yaml",
    entry: str = "train",
    resume_path: Optional[str] = None,
    devices: Optional[list[str]] = None,
):

    num_gpus = 1 if devices is None else len(devices)

    if num_gpus > 1:
        for i in range(num_gpus):
            mp.Process(
                target=proc_main,
                args=(conf_file, entry, resume_path, devices, i),
            )
    else:
        proc_main(conf_file, entry, resume_path, devices, 0)


if __name__ == "__main__":
    jsonargparse.CLI(main)
