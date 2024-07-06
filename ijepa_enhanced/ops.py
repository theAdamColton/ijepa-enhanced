import torch.nn.functional as F
import torch

MASK_ID = -100


@torch._dynamo.disable()
def masked_mean(x, m):
    return x[m].mean()


def unique_ids_inv_by_seq(ids):
    """
    ids: shape (B, S)
        unique ids along the sequence dimension, padded to form a tensor

    returns:
        unique ids, Shape: (B, U) where U is the maximum number of unique items in any sequence
        ie, U = max(len(seq.unique()) for seq in ids)

        invs: Shape (B, S) Indices that point from ids into unique ids
    """
    uids = []
    invs = []
    for seq in ids:
        u, inv = seq.unique(return_inverse=True)
        uids.append(u)
        invs.append(inv)
    pad_size = max(len(seq) for seq in uids)
    uids = [F.pad(seq, (0, pad_size - len(seq)), value=MASK_ID) for seq in uids]
    return torch.stack(uids), torch.stack(invs)


def mean_by_id(x, ids):
    """
    x: tensor of tokens with shape (B, S, Z)
    ids: tensor of shape (B, S)
        which contains sequence ids. For each sequence in the batch, the tokens of the related sequence are
        labelled with the same sequence id. Sequences across different batches can have the same sequence id,
        and not be considered part of the same sequence.

    Returns:
    tensor of shape (N, Z)
        where N is the number of total distinct sequences in the entire batch, (which are not masked padding)
    """
    device, dtype = x.device, x.dtype
    b, s, z = x.shape

    uids, inv = unique_ids_inv_by_seq(ids)
    u = uids.size(1)

    inv += (
        torch.arange(0, inv.size(0), device=device, dtype=torch.long).unsqueeze(-1) * u
    )

    inv = inv.view(-1)
    x = x.view(b * s, z)
    out = torch.zeros(b * u, z, device=device, dtype=dtype)
    out.index_reduce_(0, inv, x, reduce="mean", include_self=False)

    # get rid of padding
    uids = uids.view(-1)
    out = out[uids != MASK_ID]

    return out


def mean_by_id_ref(x, ids):
    result = []
    for sequence, sequence_ids in zip(x, ids):
        uids = sequence_ids.unique()
        for uid in uids:
            if uid == MASK_ID:
                continue
            mask = sequence_ids == uid
            result.append(sequence[mask].mean(0))
    return torch.stack(result)
