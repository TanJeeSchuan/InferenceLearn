# Shape trace: one Qwen3 block

The prompt is "The capital of France is", so T = 5. There's no batch dimension, because our engine handles one prompt until stage 4. Weight names are relative to `model.layers.0.`, so `q_proj.weight` means `W["model.layers.0.self_attn.q_proj.weight"]`. Print the shapes with the "Look around" command from lesson 0001 if you want to check them.

Fill in the Shape column with tuples like `(5, 1024)`, then check your answers:

    uv run python scripts/check_trace.py

| # | Step | Shape |
|---|------|-------|
| 1 | `x`, the block's input | |
| 2 | `h = rms_norm(x, input_layernorm.weight)` | |
| 3 | `q = h @ q_proj.weight.T` | |
| 4 | `k = h @ k_proj.weight.T` | |
| 5 | `v = h @ v_proj.weight.T` | |
| 6 | `q` split into heads of 128, `q.view(T, -1, 128)` | |
| 7 | `k` split into heads of 128 | |
| 8 | `v` split into heads of 128 | |
| 9 | `q = rms_norm(q, q_norm.weight)` | |
| 10 | `k = rms_norm(k, k_norm.weight)` | |
| 11 | `q` with heads first, `q.transpose(0, 1)` | |
| 12 | `k` with heads first | |
| 13 | `v` with heads first | |
| 14 | `q` after RoPE | |
| 15 | `k` after RoPE | |
| 16 | `k` repeated so every query head has a key head | |
| 17 | `v` repeated the same way | |
| 18 | `scores = q @ k.transpose(-2, -1) / sqrt(128)` | |
| 19 | `weights = softmax(scores + mask)` | |
| 20 | `out = weights @ v` | |
| 21 | `out` with positions first again, `out.transpose(0, 1)` | |
| 22 | `out` with heads merged, `out.reshape(T, -1)` | |
| 23 | `a = out @ o_proj.weight.T` | |
| 24 | `x = x + a` | |
| 25 | `h = rms_norm(x, post_attention_layernorm.weight)` | |
| 26 | `g = h @ gate_proj.weight.T` | |
| 27 | `u = h @ up_proj.weight.T` | |
| 28 | `m = silu(g) * u` | |
| 29 | `d = m @ down_proj.weight.T` | |
| 30 | `x = x + d`, the block's output | |
| 31 | the causal `mask` used in step 19 | |

## Questions

Answer in a sentence or two each. These get reviewed in the session, not by the script.

1. `q_proj` turns 1024 numbers per position into 2048. Where does 2048 come from?
2. In words, what does entry `[h, i, j]` of `scores` measure?
3. Which positions can position 0 see through the mask, and what does that do to row 0 of `weights`?
4. Counting steps 7 and 8, how many numbers do `k` and `v` hold together for this prompt? How many would they hold with 16 KV heads, as many as the query heads?

Answers:

1.
2.
3.
4.
