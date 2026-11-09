"""Qwen3-0.6B assembled from the ops in `ops.py`.

`W` is always the weight dict with Hugging Face's key names, e.g.
`W["model.embed_tokens.weight"]`. Print `W.keys()` and the shapes once to see
what the model is made of.
"""

from torch import Tensor

from minfer import ops


# --- Lesson 0001 -----------------------------------------------------------


def zero_layer_logits(ids: Tensor, W: dict[str, Tensor], eps: float) -> Tensor:
    """Qwen3-0.6B with all 28 transformer blocks deleted.

    embed -> final RMSNorm -> LM head.

    ids: (T,) token ids
    ->   (T, V) logits; row t scores every candidate for token t+1
    """

    embed_weight = W["model.embed_tokens.weight"]
    norm_weight = W["model.norm.weight"]
    lm_head_weight = W["lm_head.weight"]

    e = ops.embed(ids, embed_weight)
    norm_rms = ops.rms_norm(e, norm_weight, eps)
    logits = ops.lm_head(norm_rms, lm_head_weight)

    return logits
