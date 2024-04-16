import os
import torchvision
import datetime
import einx
import torch
from torch import nn
import torch.nn.functional as F

from ..utils import imread, rand_log_uniform
from ..lfq import LFQ, calculate_perplexity


def do_train(
    patch_size=8,
    c=3,
    dim=256,
    codebook_size=256,
    num_codebooks=32,
    image_file="./images/plume-512x512.jpg",
    learning_rate=1e-3,
    entropy_weight=1e2,
    entropy_sample_minimization_weight=0.0,
    entropy_batch_maximization_weight=1.0,
    commit_weight=1e-9,
    iterations=200,
    temperature=0.1,
    device="cpu",
):
    image = imread(image_file).to(device)
    patches = einx.rearrange(
        "c (nh ph) (nw pw) -> nh nw (ph pw c)", image, ph=patch_size, pw=patch_size, c=c
    )
    inproj = nn.Sequential(
        nn.Linear(patch_size * patch_size * 3, dim),
        # nn.GELU(),
    ).to(device)

    lfq = LFQ(
        dim,
        codebook_size,
        num_codebooks=num_codebooks,
        sample_minimization_weight=entropy_sample_minimization_weight,
        batch_maximization_weight=entropy_batch_maximization_weight,
        temperature=temperature,
    ).to(device)

    outproj = nn.Sequential(
        # nn.GELU(),
        nn.Linear(dim, patch_size * patch_size * c),
    ).to(device)
    optim = torch.optim.SGD(
        list(inproj.parameters()) + list(lfq.parameters()) + list(outproj.parameters()),
        lr=learning_rate,
    )
    for i in range(iterations):
        z = inproj(patches)
        z, indices, entropy_loss, commit_loss = lfq(
            z, return_losses=True, return_indices=True
        )
        xhat = outproj(z)
        loss = F.mse_loss(patches, xhat)
        loss = loss + entropy_loss * entropy_weight + commit_loss * commit_weight
        loss.backward()
        optim.step()
        optim.zero_grad()

    perplexity = calculate_perplexity(indices, codebook_size)
    xhat = xhat.clamp_(0, 1)
    loss = F.mse_loss(patches, xhat)
    xhat = einx.rearrange(
        "nh nw (ph pw c) -> c (nh ph) (nw pw)", xhat, ph=patch_size, pw=patch_size, c=c
    ).cpu()

    return loss, perplexity, xhat


def main():
    """
    use downproj, lfq, upproj to compress simple image
    """
    dirname = (
        "./test-files/test-lfq-image-compression/"
        + datetime.datetime.now().ctime()
        + "/"
    )
    os.makedirs("./test-files/test-lfq-image-compression/", exist_ok=True)
    os.makedirs(dirname)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(42)
    for _ in range(300):
        d = dict(
            learning_rate=rand_log_uniform(2e0, 1e-3),
            entropy_weight=rand_log_uniform(1e0, 1e-5),
            commit_weight=rand_log_uniform(1e0, 1e-5),
            entropy_sample_minimization_weight=rand_log_uniform(100, 1e-4),
            entropy_batch_maximization_weight=rand_log_uniform(100, 1e-4),
            temperature=rand_log_uniform(100, 0.01),
        )

        loss, perplexity, xhat = do_train(device=device, **d)

        run_string = f"loss {loss.item():.5f} perplexity {perplexity.item():.5f} "
        run_string += " ".join([f"{k}:{v:.5f}" for k, v in d.items()])

        print(run_string)

        torchvision.io.write_jpeg(
            (xhat * 255).to(torch.uint8),
            dirname + run_string + ".jpg",
            100,
        )


if __name__ == "__main__":
    main()
