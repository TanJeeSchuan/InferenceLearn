# Inference from scratch

Learning how LLM inference works by writing a Qwen3-0.6B engine (`minfer/`) from PyTorch primitives, checked against Hugging Face at every step.

Lessons: https://tanjeeschuan.github.io/InferenceLearn/

## Run

WSL2 or Linux with an NVIDIA GPU and [uv](https://docs.astral.sh/uv/):

```sh
uv sync
uv run pytest tests/test_0001_logits.py -x
```

## Layout

- `lessons/`, `reference/`, `assets/`: the course, served by GitHub Pages
- `minfer/`: the engine, written one lesson at a time
- `tests/`: each lesson's checks against Hugging Face's own model
- `MISSION.md`, `RESOURCES.md`, `NOTES.md`, `learning-records/`: course state
