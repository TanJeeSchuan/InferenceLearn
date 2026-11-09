"""The primitive operations of a transformer, written from scratch.

Rules for this file: build everything from plain tensor ops (indexing, `@`,
`exp`, `sum`, `mean`, `max`, `rsqrt`, broadcasting). Don't use `torch.nn`,
`torch.nn.functional`, or any built-in softmax / norm. The tests enforce this.
"""

import torch
from torch import Tensor


# --- Lesson 0001 -----------------------------------------------------------


def embed(ids: Tensor, E: Tensor) -> Tensor:
    """Look up one embedding row per token id.

    ids: (T,) int64 token ids
    E:   (V, D) embedding table
    ->   (T, D)
    """
    return E[ids]


def rms_norm(x: Tensor, w: Tensor, eps: float) -> Tensor:
    """Divide each row of x by its root-mean-square, then scale by w.

    x:   (..., D)
    w:   (D,) learned per-dimension scale
    eps: added inside the square root, so a row of zeros doesn't divide by zero
    ->   (..., D), same dtype as x
    """
    return (x / (torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps))) * w


def lm_head(x: Tensor, E: Tensor) -> Tensor:
    """Score every vocabulary entry against each position's final vector.

    x: (T, D) normed final hidden states
    E: (V, D) the same table `embed` used (Qwen3-0.6B ties these weights)
    -> (T, V) logits
    """

    return x @ E.T


def softmax(z: Tensor) -> Tensor:
    """Turn logits into probabilities along the last dimension.

    z: (..., V) any real numbers, including very large ones
    -> (..., V) non-negative, each row sums to 1, never NaN or inf
    """

    shifted = z - z.max(dim=-1, keepdim=True).values          # z minus its row max (keepdim!)
    e = shifted.exp()
    total = e.sum(-1, keepdim=True)

    return e / total
