import einx
import safetensors
import safetensors.torch
import torch
import hydra
import os
import subprocess

from ijepa_enhanced.transformer import TransformerBlock

from ..vit import ViT
from ..predictor import Predictor


def copy_in_coerce_(tgt: torch.Tensor, src: torch.Tensor):
    assert tgt.ndim == src.ndim
    new_shape = []
    for i in range(src.ndim):
        stop = min(tgt.shape[i], src.shape[i])
        new_shape.append(slice(stop))

    tgt[new_shape].copy_(src[new_shape])


def filter_remove_prefix(d, pre):
    d = {k.removeprefix(pre): v for k, v in d.items() if k.startswith(pre)}
    return d


def copy_in_weight_bias_(tgt, src):
    copy_in_coerce_(getattr(tgt, "weight"), src["weight"])
    copy_in_coerce_(getattr(tgt, "bias"), src["bias"])
    return tgt


def copy_in_block_(tgt: TransformerBlock, src):
    norm1_src = filter_remove_prefix(src, "norm1.")
    copy_in_weight_bias_(tgt.norm1, norm1_src)

    qkv_src = filter_remove_prefix(src, "attn.qkv.")
    copy_in_weight_bias_(tgt.attn.qkv, qkv_src)

    attn_proj_src = filter_remove_prefix(src, "attn.proj.")
    copy_in_weight_bias_(tgt.attn.proj, attn_proj_src)

    norm2_src = filter_remove_prefix(src, "norm2.")
    copy_in_weight_bias_(tgt.norm2, norm2_src)

    mlp_fc1_src = filter_remove_prefix(src, "mlp.fc1.")
    copy_in_weight_bias_(tgt.mlp.fc1, mlp_fc1_src)

    mlp_fc2_src = filter_remove_prefix(src, "mlp.fc2.")
    copy_in_weight_bias_(tgt.mlp.fc2, mlp_fc2_src)

    return tgt


def merge_pretrained_ijepa_vit(
    model_path="https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.16-448px-300e.pth.tar",
    model_dir="./pretrained-models/",
    vit: ViT = None,
    predictor: Predictor = None,
):
    vit.requires_grad_(False)
    predictor.requires_grad_(False)

    model_name = os.path.basename(model_path)
    model_save_path = os.path.join(model_dir, model_name)

    if not os.path.exists(model_save_path):
        os.makedirs(model_dir, exist_ok=True)
        subprocess.run(["wget", model_path, "-O", model_save_path])

    print("loading state dict", model_save_path)
    state_dict = torch.load(model_save_path, map_location="cpu")

    predictor_src = state_dict["predictor"]

    predictor_src = filter_remove_prefix(predictor_src, "module.")

    copy_in_coerce_(predictor.is_prediction_token, predictor_src["mask_token"][0, 0])
    copy_in_weight_bias_(
        predictor.proj_in, filter_remove_prefix(predictor_src, "predictor_embed.")
    )

    for i, block_tgt in enumerate(predictor.transformer.layers):
        block_src = filter_remove_prefix(predictor_src, f"predictor_blocks.{i}.")
        copy_in_block_(block_tgt, block_src)

    copy_in_weight_bias_(
        predictor.norm, filter_remove_prefix(predictor_src, "predictor_norm.")
    )
    copy_in_weight_bias_(
        predictor.pred_head, filter_remove_prefix(predictor_src, "predictor_proj.")
    )

    # vit encoder

    encoder_src = state_dict["encoder"]

    encoder_src = filter_remove_prefix(encoder_src, "module.")

    patch_emb_src = filter_remove_prefix(encoder_src, "patch_embed.proj.")
    patch_emb_src["weight"] = einx.rearrange(
        "z c h w -> z (c h w)", patch_emb_src["weight"]
    )
    copy_in_weight_bias_(
        vit.to_patch_embedding[1],
        patch_emb_src,
    )
    for i, block_tgt in enumerate(vit.transformer.layers):
        block_src = filter_remove_prefix(encoder_src, f"blocks.{i}.")
        copy_in_block_(block_tgt, block_src)
    copy_in_weight_bias_(
        vit.norm,
        filter_remove_prefix(encoder_src, "norm."),
    )

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
