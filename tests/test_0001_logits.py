"""Lesson 0001: tokens -> logits. Run: uv run pytest tests/test_0001_logits.py -x"""

import pytest
import torch
from check import assert_close

from minfer import ops
from minfer.model import zero_layer_logits


def _zero_layer_ref(hf, ids):
    return hf.lm_head(hf.model.norm(hf.model.embed_tokens(ids)))


def test_embed(hf, W, ids, forbid):
    ref = hf.model.embed_tokens(ids)
    forbid("embedding")
    mine = ops.embed(ids, W["model.embed_tokens.weight"])
    assert_close("embed", mine, ref, atol=0, rtol=0, hint="row i of the output is row ids[i] of E")


@pytest.mark.parametrize(
    ("scale", "hint"),
    [
        (1.0, "mean of squares over the last dim, keepdim so it broadcasts back over D"),
        (1e3, "large inputs: the output should not depend on the input's scale"),
        (1e-4, "only failing on tiny inputs? eps belongs inside the square root"),
    ],
    ids=["real", "huge", "tiny"],
)
def test_rms_norm(hf, W, ids, forbid, scale, hint):
    x = hf.model.embed_tokens(ids) * scale
    ref = hf.model.norm(x)
    forbid("rms_norm")
    mine = ops.rms_norm(x, W["model.norm.weight"], hf.config.rms_norm_eps)
    assert_close(f"rms_norm(scale={scale:g})", mine, ref, atol=1e-6, rtol=1e-5, hint=hint)


def test_lm_head(hf, W, ids, forbid):
    x = hf.model.norm(hf.model.embed_tokens(ids))
    ref = hf.lm_head(x)
    forbid("linear")
    mine = ops.lm_head(x, W["model.embed_tokens.weight"])
    assert_close("lm_head", mine, ref, atol=1e-4, rtol=1e-4, hint="(T, D) against (V, D): one of them needs transposing")


def test_softmax_real_logits(hf, ids, forbid):
    z = _zero_layer_ref(hf, ids)
    ref = torch.softmax(z, dim=-1)
    forbid("softmax")
    assert_close("softmax(logits)", ops.softmax(z), ref, atol=1e-6, rtol=1e-4, hint="normalize along the last dim")


def test_softmax_huge_logits(device, forbid):
    z = torch.tensor([[1000.0, 1001.0, 1002.0], [-1000.0, 0.0, 1000.0]], device=device)
    ref = torch.softmax(z, dim=-1)
    forbid("softmax")
    assert_close(
        "softmax(huge)",
        ops.softmax(z),
        ref,
        atol=1e-6,
        rtol=1e-5,
        hint="exp(1000) is inf in fp32. softmax(z) == softmax(z - c) for any constant c per row",
    )


def test_softmax_leading_dims(device, forbid):
    z = torch.randn(2, 3, 7, device=device)
    ref = torch.softmax(z, dim=-1)
    forbid("softmax")
    assert_close("softmax(2,3,7)", ops.softmax(z), ref, atol=1e-6, rtol=1e-5, hint="dim=-1 and keepdim=True")


def test_zero_layer_logits(hf, W, ids, forbid):
    ref = _zero_layer_ref(hf, ids)
    forbid("embedding", "rms_norm", "linear")
    mine = zero_layer_logits(ids, W, hf.config.rms_norm_eps)
    assert_close("zero_layer_logits", mine, ref, atol=1e-4, rtol=1e-4, hint="embed -> rms_norm(model.norm.weight) -> lm_head")
