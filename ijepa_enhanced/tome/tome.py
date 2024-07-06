from typing import Optional
import einx
import torch


def expand_trailing(y, x):
    """
    x has more dimensions than y

    unsqueezes y to match the number of dimensions of x
    and then expands (repeats) y along these trailing dimensions
    so that the shapes of x and y match
    """
    y_shape = y.shape
    x_shape = x.shape
    new_y_shape = list(y_shape) + list(x_shape[len(y_shape) :])
    for _ in range(len(new_y_shape) - len(y_shape)):
        y = y.unsqueeze(-1)
    y = y.expand(new_y_shape)
    return y


def merge_all(x: torch.Tensor, adm: torch.Tensor) -> torch.Tensor:
    """
    composes multiple merges all at once
    Allows you to merge a original tensor in a single step rather than
    using tm.merge() however many times merging was applied.
    """
    b, s, *_ = x.shape
    assert s == adm.size(
        -1
    ), "x needs to have the same sequence length as the original input"
    normalized_adm = adm / adm.sum(-1, keepdim=True)
    return einx.dot("b s2 s1, b s1 ... -> b s2 ...", normalized_adm, x)


def unmerge_all(x: torch.Tensor, adm: torch.Tensor) -> torch.Tensor:
    """
    x shape: (batch, sequence_length - r, ...)

    Undo one or more TokenMerger.merge calls

    returns a tensor of shape: (batch, original_sequence_length, ...)
    """

    unmerge_indices = adm.argmax(dim=1)
    return x.gather(dim=1, index=expand_trailing(unmerge_indices, x))


class TokenMerger:
    """
    Merges and merges tokens, exploiting the token to token similarity in k.

    One TokenMerger instance represents a single merge layer, with a fixed input and
    output sequence length.

    Example using a single TokenMerger layer:
        >>> batch_size = 32
        >>> sequence_length = 128
        >>> hidden_size = 768
        >>> r = 32
        >>> keys = torch.randn(batch_size, sequence_length, hidden_size)
        >>> tm = TokenMerger(keys, r)
        >>> x = torch.randn(batch_size, sequence_length, 64)
        >>> merged_x = tm(x) # shape: batch_size, sequence_length - r, 64
        >>> unmerged_x = tm.unmerge(merged_x) # shape: batch_size, sequence_length, 64

    If you want to apply TokenMerger over multiple layers and you want to unmerge the
    final tokens into the shape of the original tokens, you will need to pass the TokenMerger.adm
    from layer to layer.

    This looks like this:
        >>> x = torch.randn(1, 16, 2)
        >>> tm1 = TokenMerger(x, 4)
        >>> x_merged1 = tm1(x) # shape: (1, 12, 2)
        >>> tm2 = TokenMerger(x_merged1, 4, adm=tm1.adm) # pass adm to tm2
        >>> x_merged2 = tm2(x_merged1) # shape: (1, 8, 2)
        >>> rec_x = tm2.unmerge(x_merged2) # shape: (1, 16, 2)

    TokenMerger also accepts sequence_ids and mask_id as optional parameters.
    These parameters are meant for specifying the distinct datapoints when using sequence packing.
    Each sequence in the batch can contain more than one datapoint. There might also be padding.
    Per sequence, each datapoint is identified using a unique integer id.
    For example, sequence_ids[0] could equal [0, 0, 0, 1, 1,  -100]. sequence_ids[0] has 2
    datapoints, identified by 0 and 1. Datapoint 0 has 3 tokens, datapoint 1 has 2 tokens. There
    is also one padding token.
    TokenMerger will not merge tokens from different datapoints. Datapoint 0 and datapoint 1 will
    never be merged. TokenMerger will merge padding tokens only with other padding tokens.
    """

    @torch.no_grad()
    def __init__(
        self,
        k: torch.Tensor,
        r: int,
        adm=None,
        sequence_ids=None,
        mask_id=-100,
    ):
        """
        k: token embeddings, can be the key values from attention.
            shape: batch, sequence length, z

        r: the number of tokens to remove.

        adm: Optional torch bool tensor of shape (b, sequence length, original sequence length)
            Pass this value if needing to unmerge multiple merge steps all at once.
            This is a adjacency matrix, where the rows (dim=-2) represent the merged tokens,
            and each column (dim=-1) represents the original tokens. For example, if
            the zeroeth row contains [False, False, False, True, True], it means that
            token 0, before being merged by this TokenMerger, is composed of the
            tokens 3 and 4 (counting from zero).

        sequence_ids: LongTensor of shape (b, sequence_length)
            Uniquely identifies each element of each sequence.
            Protects tokens that are not part of the same instance from being merged.
            Optional
            shape: batch, sequence length

        mask_id: integer id of the sequence id that is considered a part of the sequence padding.
        """
        sequence_length = k.shape[1]
        assert sequence_length // 2 >= r
        assert r > 0
        k = k / k.norm(dim=-1, keepdim=True)
        # step 1.) Assign tokens to set A or set B
        a, b = k[:, ::2], k[:, 1::2]
        # step 2.) Draw one edge between each token in set A and the most similar token in set B
        scores = torch.matmul(
            a, b.movedim(1, 2)
        )  # einx.dot("b s1 z, b s2 z -> b s1 s2", a, b)

        # masks out scores
        if sequence_ids is not None:
            a_ids, b_ids = sequence_ids[:, ::2], sequence_ids[:, 1::2]

            attention_mask = a_ids.unsqueeze(2) == b_ids.unsqueeze(1)

            # scores where ids are not equal should be -inf
            scores.masked_fill_(~attention_mask, torch.finfo(scores.dtype).min)
            pad_mask = (a_ids == mask_id).unsqueeze(2) & (b_ids == mask_id).unsqueeze(1)
            # makes pad tokens have high scores
            scores.masked_fill_(pad_mask, torch.finfo(scores.dtype).max)

        node_max, node_idx = scores.max(2)  # einx.max("b s1 s2 -> b s1", scores)
        edge_idx = node_max.argsort(dim=-1, descending=True)  # shape: b s 1
        # step 3.) Keep the top r most similar edges
        unm_idx = edge_idx[:, r:]  # Unmerged Tokens
        src_idx = edge_idx[:, :r]  # Merged Tokens
        dst_idx = node_idx.gather(dim=1, index=src_idx)  # shape: b r 1
        self.unm_idx = unm_idx
        self.src_idx = src_idx
        self.dst_idx = dst_idx
        self.unmerged_sequence_length = k.shape[1]
        self.merged_sequence_length = unm_idx.size(1) + b.size(1)
        self.r = r
        self.mask_id = mask_id

        if sequence_ids is not None:
            # Merges ids, but doesn't do any reduction on merged ids,
            # Simply takes the items from set b.
            # The reduction is uncessary because each set of merged ids is assumed to be identical,
            # meaning that ids are not merged with ids that are different.
            # An id 0 shouldn't be be merged with any id that is not 0.
            unm_ids = a_ids.gather(dim=1, index=unm_idx)
            a_ids = a_ids.gather(dim=1, index=src_idx)
            assert torch.equal(
                b_ids,
                b_ids.scatter(dim=1, index=dst_idx, src=a_ids),
            ), "These ids should be equal. If this test fails it means that attention mask was not properly computed and or tokens were incorrectly merged over sequence-id boundaries."
            self.merged_ids = torch.cat((unm_ids, b_ids), 1)
        else:
            self.merged_ids = None

        if adm is None:
            adm = torch.eye(sequence_length, device=k.device)
            adm = adm.unsqueeze(0).repeat(
                k.size(0), 1, 1
            )  # einx.rearrange("s1 s2 -> b s1 s2", adm, b=k.size(0))
        else:
            assert adm.size(0) == k.size(0)
            assert adm.size(1) == sequence_length
        self.adm = self.merge(adm, "amax")

    def __call__(self, x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        return self.merge(x, mode)

    def chain(self, k: torch.Tensor, r: Optional[int] = None):
        prev = self
        next = TokenMerger(
            k, prev.r if r is None else r, prev.adm, prev.merged_ids, prev.mask_id
        )
        return next

    def merge(self, x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        """
        x shape: (batch, sequence length, ...)

        Does bipartite merging as keyed by unm_idx and src_idx
        """

        assert x.size(1) == self.unmerged_sequence_length

        a, b = x[:, ::2], x[:, 1::2]
        unm = a.gather(dim=1, index=expand_trailing(self.unm_idx, a))
        a = a.gather(dim=1, index=expand_trailing(self.src_idx, a))

        og_dtype = b.dtype
        b = b.float()
        a = a.float()

        # step 4.) Merge connected tokens

        if mode == "mlerp":
            b = b.scatter_reduce(
                dim=1, index=expand_trailing(self.dst_idx, b), src=a, reduce="mean"
            )
            b_norm = b.norm(p=2, dim=-1)
            n = b_norm.scatter_reduce(
                dim=1,
                index=self.dst_idx,
                src=a.norm(p=2, dim=-1),
                reduce="amax",
            )
            b = (b / b_norm.unsqueeze(-1)) * n.unsqueeze(-1)
        elif mode == "drop":
            pass
        else:
            b = b.scatter_reduce(
                dim=1,
                index=expand_trailing(self.dst_idx, b),
                src=a,
                reduce=mode,
            )

        b = b.to(og_dtype)

        # step 5.) Concatenate sets back together
        return torch.cat((unm, b), 1)

    def unmerge_all(self, x: torch.Tensor) -> torch.Tensor:
        return unmerge_all(x, self.adm)

    def merge_all(self, x):
        return merge_all(x, self.adm)
