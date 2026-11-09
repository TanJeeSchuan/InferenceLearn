# Mission: LLM inference, by writing an inference engine

## Why
Understand what actually happens between a prompt and a generated token, from the tensor math up to the scheduling and memory tricks that serving engines use, well enough to read vLLM or llama.cpp source and know why each part exists. Long-term goal: land a real PR in one of those engines.

## Success looks like
- My own Qwen3-0.6B forward pass, written from PyTorch primitives, matches Hugging Face logits on my RTX 4050.
- I can explain with roofline arithmetic why decode is memory-bound and what batching buys, then measure it.
- I have built continuous batching, a paged KV cache, speculative decoding and weight-only quantization into that engine, each with a design doc I wrote before reading how the experts did it.
- I have written a Triton attention kernel that beats my PyTorch version.
- I can open the vLLM or llama.cpp codebase and find my way to the part matching each stage I built.

## Constraints
- Hobby pace, no deadline. Lessons are 30 to 60 minutes.
- Hardware: RTX 4050 laptop GPU (6 GB VRAM) on Win11 + WSL2. One GPU, so no multi-GPU topics.
- Background: CS fresh grad. Comfortable with matmul shapes. Softmax, attention and KV cache start from zero. CUDA from coursework, mostly forgotten.

## Out of scope
- Training, backprop, fine-tuning.
- Multi-GPU and tensor/pipeline parallelism.
- HTTP serving, OpenAI-compatible APIs, deployment.
- Choosing the upstream project (vLLM/SGLang vs llama.cpp): deferred until after stage 5.
