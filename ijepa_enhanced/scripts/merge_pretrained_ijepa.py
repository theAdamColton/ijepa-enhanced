import safetensors
import safetensors.torch
import torch
import hydra
import os
import subprocess
from torch.nn import functional as F

from ..vit import ViT
from ..predictor import Predictor


def copy_in_coerce_(tgt: torch.Tensor, src: torch.Tensor):
    ndim = src.ndim
    assert ndim < 3
    assert tgt.ndim == src.ndim
    shape = tgt.shape
    to_expand = 2
    for _ in range(to_expand):
        src = src.unsqueeze(0)
    src = F.interpolate(
        src,
        shape,
    )
    for _ in range(to_expand):
        src = src.squeeze(0)
    tgt.copy_(src)


def merge_pretrained_ijepa_vit(
    model_path="https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.16-448px-300e.pth.tar",
    model_dir="./pretrained-models/",
    vit: ViT = None,
    predictor: Predictor = None,
):
    model_name = os.path.basename(model_path)
    model_save_path = os.path.join(model_dir, model_name)

    if not os.path.exists(model_save_path):
        os.makedirs(model_dir, exist_ok=True)
        subprocess.run(["wget", model_path, "-O", model_save_path])

    print("loading state dict", model_save_path)
    state_dict = torch.load(model_save_path, map_location="cpu")

    predictor_src = state_dict["predictor"]
    predictor_tgt = predictor.state_dict()
    predictor_tgt_depth = predictor.transformer.depth

    predictor_src = {k.removeprefix("module."): v for k, v in predictor_src.items()}

    for k, v in predictor_src.items():
        if k.startswith("predictor_blocks"):
            k = k.removeprefix("predictor_blocks.")
            i = int(k[: k.find(".")])

            if i >= predictor_tgt_depth:
                continue

            key_layer = f"transformer.layers.{i}."

            if "norm1.weight" in k:
                copy_in_coerce_(predictor_tgt[key_layer + "norm1.weight"], v)

            elif "norm1.bias" in k:
                copy_in_coerce_(predictor_tgt[key_layer + "norm1.bias"], v)

            elif "qkv.weight" in k:
                copy_in_coerce_(predictor_tgt[key_layer + "attn.qkv.weight"], v)

            elif "attn.proj.weight" in k:
                copy_in_coerce_(predictor_tgt[key_layer + "attn.proj.weight"], v)

            elif "norm2.weight" in k:
                copy_in_coerce_(predictor_tgt[key_layer + "norm2.weight"], v)

            elif "norm2.bias" in k:
                copy_in_coerce_(predictor_tgt[key_layer + "norm2.bias"], v)

            elif "mlp.fc1.weight" in k:
                copy_in_coerce_(predictor_tgt[key_layer + "mlp.layers.0.weight"], v)

            elif "mlp.fc2.weight" in k:
                copy_in_coerce_(predictor_tgt[key_layer + "mlp.layers.2.weight"], v)

            print("copied pred ", k)

    copy_in_coerce_(
        predictor_tgt["norm.weight"], predictor_src["predictor_norm.weight"]
    )
    copy_in_coerce_(predictor_tgt["norm.bias"], predictor_src["predictor_norm.bias"])
    copy_in_coerce_(
        predictor_tgt["pred_head.weight"], predictor_src["predictor_proj.weight"]
    )

    predictor.load_state_dict(predictor_tgt)

    # vit encoder

    encoder_src = state_dict["encoder"]
    encoder_tgt = vit.state_dict()
    encoder_tgt_depth = vit.model.depth

    encoder_src = {k.removeprefix("module."): v for k, v in encoder_src.items()}

    for k, v in encoder_src.items():
        if k.startswith("blocks."):
            k = k.removeprefix("blocks.")
            i = int(k[: k.find(".")])

            if i >= encoder_tgt_depth:
                continue

            key_layer = f"model.layers.{i}."

            if "norm1.weight" in k:
                copy_in_coerce_(encoder_tgt[key_layer + "norm1.weight"], v)

            elif "norm1.bias" in k:
                copy_in_coerce_(encoder_tgt[key_layer + "norm1.bias"], v)

            elif "qkv.weight" in k:
                copy_in_coerce_(encoder_tgt[key_layer + "attn.qkv.weight"], v)

            elif "attn.proj.weight" in k:
                copy_in_coerce_(encoder_tgt[key_layer + "attn.proj.weight"], v)

            elif "norm2.weight" in k:
                copy_in_coerce_(encoder_tgt[key_layer + "norm2.weight"], v)

            elif "norm2.bias" in k:
                copy_in_coerce_(encoder_tgt[key_layer + "norm2.bias"], v)

            elif "mlp.fc1.weight" in k:
                copy_in_coerce_(encoder_tgt[key_layer + "mlp.layers.0.weight"], v)

            elif "mlp.fc2.weight" in k:
                copy_in_coerce_(encoder_tgt[key_layer + "mlp.layers.2.weight"], v)

            else:
                continue

            print("copied encoder", k)

    vit.load_state_dict(encoder_tgt)

    return vit, predictor


@hydra.main(version_base=None, config_path="../../conf", config_name="conf")
def main(config):
    vit = ViT(**config.model.vit)
    predictor = Predictor(**config.model.predictor)
    vit, predictor = merge_pretrained_ijepa_vit(
        vit=vit, predictor=predictor, model_dir=config.pretrained_model_path
    )
    safetensors.torch.save_model(
        vit,
        f"{config.pretrained_model_path}/{config.model.vit.name}.safetensors",
    )

    safetensors.torch.save_model(
        predictor,
        f"{config.pretrained_model_path}/{config.model.predictor.name}.safetensors",
    )

    print("saved merged models")


if __name__ == "__main__":
    main()
