# Teaching notes

## Method (agreed 2026-11-09)
- Stages 1 to 3 (model math): read the docs, then write a shape trace (every tensor's shape through one block), which gets reviewed against the HF config, then code.
- Stages 4 to 8 (engine): I state the problem plus a bottleneck measured on the learner's own code. The learner writes a 1-page design in `designs/stage-N.md` without reading anything, then reads the primary source and writes a "their design vs mine" diff, then codes. The learner opted into productive failure.
- Feedback comes from pytest comparisons against Hugging Face `transformers` outputs. The oracle is always the real model, never a re-statement of the expected answer.
- Every lesson opens with a 3 to 5 question spaced-review quiz over earlier lessons.
- Hints come in steps, only the step asked for: concept, then shapes/pseudocode, then the line. Never write solution files unless the learner says "show me".

## Spine
1 forward pass, 2 sampling, 3 KV cache + roofline, 4 continuous batching, 5 paged KV, 6 speculative decoding, 7 int8/int4 weight-only quant, 8 Triton kernels (fused RMSNorm, FlashAttention-style). Model: Qwen3-0.6B only.

## Setup
- Code runs on the home machine (Win11 + WSL2, RTX 4050 6 GB), linked to the T3 session. The repo is cloned inside the WSL filesystem, not `/mnt/c`.
- The server that authors the course has ~7 GB RAM shared with other services: never load the model there.
- Public repo, lessons on GitHub Pages.
- Tests run in fp32 to keep tolerances tight. bf16 behavior is its own topic later (stage 3/7).
- Laptop facts (measured 2026-11-09): WSL distro Ubuntu-26.04, repo at `~/InferenceLearn`, uv 0.13.7, torch 2.11.0+cu128, transformers 5.22.2, Python 3.12. The Windows desktop holds ~1.6 GB of VRAM; the lesson 0001 tests peak at 2.3 GB allocated. WSL's `/tmp` gets wiped between separate `wsl` calls, so keep scratch files in `~/.scratch`.
- Verifying a lesson: delegate to the laptop and put the reference solution only in a throwaway clone (`~/.scratch/minfer-verify`). Check that the reference passes, and that a planted bug makes each test fail.

## Preferences
- Wants to be challenged. Terse prose, no filler.
- Every formula comes with its PyTorch line and a 3-4 element numeric example. New symbols go into `reference/notation.html` (see LR 0002).
- Upstream target is decided after stage 5 with a short lesson comparing the codebases.
