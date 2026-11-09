"""Lesson 0001 demo: what does Qwen3 predict with 0 layers vs all 28?

uv run python scripts/zero_layer.py "The capital of France is"
"""

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from minfer.model import zero_layer_logits
from minfer.ops import softmax

MODEL_ID = "Qwen/Qwen3-0.6B"
torch.set_grad_enabled(False)


def top(tok, p, k=3):
    vals, idx = p.topk(k)
    return "  ".join(f"{tok.decode(i)!r:<10} {v:.2f}" for v, i in zip(vals.tolist(), idx.tolist()))


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    hf = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(dev).eval()
    ids = tok(prompt, return_tensors="pt").input_ids[0].to(dev)

    p0 = softmax(zero_layer_logits(ids, dict(hf.state_dict()), hf.config.rms_norm_eps))
    p28 = softmax(hf(ids[None]).logits[0])

    print(f"{'after token':<14}| {'0 layers (yours)':<48}| 28 layers (HF)")
    for t in range(len(ids)):
        print(f"{tok.decode(ids[t])!r:<14}| {top(tok, p0[t]):<48}| {top(tok, p28[t])}")


if __name__ == "__main__":
    main()
