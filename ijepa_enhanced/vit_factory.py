import einx
import torch

from .vit import ViT


def copy_in_coerce_(tgt: torch.Tensor, src: torch.Tensor):
    assert tgt.ndim == src.ndim
    new_shape = []
    for i in range(src.ndim):
        stop = min(tgt.shape[i], src.shape[i])
        new_shape.append(slice(stop))

    tgt[new_shape].copy_(src[new_shape])
    src._did_merge = True
    tgt._did_merge = True


def copy_in_weight_bias_(tgt, src):
    copy_in_coerce_(tgt.weight, src.weight)
    if hasattr(tgt, "bias") and tgt.bias is not None:
        copy_in_coerce_(tgt.bias, src.bias)


def copy_in_block_(tgt, src):
    copy_in_weight_bias_(tgt.norm1, src.norm1)
    copy_in_weight_bias_(tgt.attn.qkv, src.attn.qkv)
    copy_in_weight_bias_(tgt.attn.proj, src.attn.proj)
    copy_in_coerce_(tgt.ls1.gamma, src.ls1.gamma)
    copy_in_weight_bias_(tgt.norm2, src.norm2)
    copy_in_weight_bias_(tgt.mlp.fc1, src.mlp.fc1)
    copy_in_weight_bias_(tgt.mlp.fc2, src.mlp.fc2)
    copy_in_coerce_(tgt.ls2.gamma, src.ls2.gamma)


@torch.no_grad()
def merge(vit: ViT):

    assert vit.use_layerscale
    assert vit.use_joint_position_embed
    vit_src = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

    pos_embed_src = vit_src.pos_embed[0, 1:]
    copy_in_coerce_(vit.position_embed.embed, pos_embed_src)
    vit_src.pos_embed._did_merge = True

    patch_embed_weight = vit_src.patch_embed.proj.weight
    patch_embed_bias = vit_src.patch_embed.proj.bias
    copy_in_coerce_(
        vit.patch_embed.weight,
        einx.rearrange("z c ph pw -> z (c ph pw)", patch_embed_weight),
    )
    vit_src.patch_embed.proj.weight._did_merge = True
    copy_in_coerce_(vit.patch_embed.bias, patch_embed_bias)

    for tgt, src in zip(vit.blocks.blocks, vit_src.blocks):
        copy_in_block_(tgt, src)

    copy_in_weight_bias_(vit.norm, vit_src.norm)

    for n, p in vit_src.named_parameters():
        if not hasattr(p, "_did_merge"):
            print("predictor unmerged from src: ", n)
    for n, p in vit.named_parameters():
        if not hasattr(p, "_did_merge"):
            print("predictor unmerged from tgt: ", n)

    return vit
