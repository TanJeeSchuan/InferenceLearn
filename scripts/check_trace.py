"""Check a lesson's shape trace without printing the answers.

uv run python scripts/check_trace.py [traces/0002-block.md]

The answer key is stored as hashes, so reading this file doesn't give the shapes away.
"""

import hashlib
import re
import sys

KEYS = {
    "0002": {
        1: "b17bda4bea457ccb", 2: "b1d3e9748a1211ff", 3: "f508632675df258d", 4: "aa0a63c47eea7fef",
        5: "69d65d4675241e1f", 6: "b6798b52498b8d3f", 7: "b82cfc39c178b32f", 8: "3a058e04383271c8",
        9: "e68069cafbfa10e7", 10: "fafea87b633f4e84", 11: "288214cee72767e8", 12: "a569edbfba26a09d",
        13: "9051c89945dd1980", 14: "32f0133a69b3185b", 15: "82806f38c8b31609", 16: "f96b33a157e3b1cb",
        17: "3b95539a5328ee7f", 18: "bee4ba1b1a8c5da6", 19: "fffdf03306cf8876", 20: "cb8bfc543343bf12",
        21: "780ab3029b47746b", 22: "69fd27d79a355a66", 23: "799df9a234c9cb41", 24: "99472f8d0e461c13",
        25: "216759eed5b862ed", 26: "800a63b2dd2a2cb7", 27: "c463b55266fbbc5d", 28: "5c7dc73cfca28e07",
        29: "1e6121f35045139a", 30: "a29404e6c2d0f0f5", 31: "a92660f6f57ec5fd",
    },
}

HINTS = {
    "0002": {
        1: "the block's input is the residual stream: (T, hidden_size)",
        2: "RMSNorm never changes the shape",
        3: "q_proj.weight is (out_features, in_features); out_features = num_attention_heads x head_dim",
        4: "k_proj's out_features = num_key_value_heads x head_dim",
        5: "v_proj has the same shape as k_proj",
        6: "split each position's numbers into chunks of head_dim; the -1 is how many chunks fit",
        7: "k has fewer heads than q",
        8: "v is split like k",
        9: "q_norm normalizes over the last dim (head_dim) and keeps the shape",
        10: "k_norm keeps the shape too",
        11: "transpose(0, 1) swaps the first two dims",
        12: "same swap as q",
        13: "same swap as q",
        14: "RoPE rotates pairs of numbers inside each vector; the shape stays the same",
        15: "RoPE keeps the shape",
        16: "each KV head serves num_attention_heads / num_key_value_heads query heads",
        17: "v is repeated like k",
        18: "batched matmul over heads: (heads, T, 128) @ (heads, 128, T)",
        19: "adding the mask and softmax both keep the shape",
        20: "batched matmul: (heads, T, T) @ (heads, T, 128)",
        21: "swap the first two dims back",
        22: "merge the last two dims into one",
        23: "o_proj.weight is (1024, 2048)",
        24: "a residual add needs both sides to have the same shape",
        25: "RMSNorm never changes the shape",
        26: "gate_proj's out_features is intermediate_size",
        27: "up_proj matches gate_proj",
        28: "silu and * both work elementwise",
        29: "down_proj.weight is (1024, 3072)",
        30: "the block's output has the same shape as its input",
        31: "one entry per (query position, key position) pair; the same mask serves every head",
    },
}

ROW = re.compile(r"^\|\s*(\d+)\s*\|.*\|\s*(.*?)\s*\|\s*$")


def digest(lesson: str, row: int, dims: list[str]) -> str:
    return hashlib.sha256(f"{lesson}:{row}:{','.join(dims)}".encode()).hexdigest()[:16]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "traces/0002-block.md"
    lesson = re.search(r"(\d{4})", path).group(1)
    key, hints = KEYS[lesson], HINTS[lesson]

    answers = {}
    for line in open(path, encoding="utf-8"):
        m = ROW.match(line)
        if m and int(m.group(1)) in key:
            answers[int(m.group(1))] = m.group(2)

    right = 0
    for row in sorted(key):
        dims = re.findall(r"\d+", answers.get(row, ""))
        if not dims:
            print(f"  {row:>2}  blank")
        elif digest(lesson, row, dims) == key[row]:
            right += 1
            print(f"  {row:>2}  ok")
        else:
            print(f"  {row:>2}  wrong   hint: {hints[row]}")
    print(f"\n{right} / {len(key)} correct")
    sys.exit(0 if right == len(key) else 1)


if __name__ == "__main__":
    main()
