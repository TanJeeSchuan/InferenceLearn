# LLM Inference Resources

Every link below was fetched on 2026-11-09 and the title and authors checked against the page. The "Under the tensor library" section was fetched on 2026-11-13.

## Knowledge

### Model and forward pass (Stage 1)
- [Paper: Qwen3 Technical Report by Qwen Team (2025)](https://arxiv.org/abs/2505.09388)
  Section 2 and Table 1 define the dense architecture: GQA, SwiGLU, RoPE, pre-norm RMSNorm, no QKV bias, added QK-Norm; 0.6B has 28 layers, 16 Q / 8 KV heads, tied embeddings, 151,669-token vocab. Use for: the spec your forward pass must match.
- [Model card: Qwen/Qwen3-0.6B by Qwen Team (2025)](https://huggingface.co/Qwen/Qwen3-0.6B)
  Weights, config.json, tokenizer, chat template with the `enable_thinking` switch, and a Best Practices section with recommended sampling parameters. Use for: loading weights in stage 1, picking default sampler settings in stage 2.
- [Source: modeling_qwen3.py by Hugging Face and the Qwen team (2025)](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py)
  The reference PyTorch implementation, generated from `modular_qwen3.py` in the same folder. Use for: diffing your logits layer by layer against a known-good model; read `Qwen3Attention` for where QK-Norm sits relative to RoPE.
- [Code: Qwen3 From Scratch (LLMs-from-scratch ch05/11_qwen3) by Sebastian Raschka (2025)](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/11_qwen3)
  Standalone notebooks implementing Qwen3 0.6B in plain PyTorch, plus a second notebook that adds a KV cache. Use for: checking your work after you have attempted stage 1 yourself; it reads linearly where the HF file is layered with abstractions.
- [Video + text: Transformers, the tech behind LLMs (Deep Learning ch. 5) by Grant Sanderson, 3Blue1Brown (2024)](https://www.3blue1brown.com/lessons/gpt)
  Visual walkthrough of tokens, embeddings, unembedding and softmax with temperature. Use for: lesson 0001 intuition; the same chapter series covers attention next.
- [Paper: A Mathematical Framework for Transformer Circuits by Elhage et al., Anthropic (2021)](https://transformer-circuits.pub/2021/framework/index.html)
  The "Zero-Layer Transformers" section shows an embed then unembed model can only learn bigram statistics. Use for: explaining what the zero-layer model in lesson 0001 predicts and why.
- [Video + text: Attention in transformers, step-by-step (Deep Learning ch. 6) by Grant Sanderson, 3Blue1Brown (2024)](https://www.3blue1brown.com/lessons/attention)
  Queries, keys, values, the causal mask and multiple heads, drawn. Use for: lesson 0002 intuition before the shape trace.
- [Paper: Dissecting Recall of Factual Associations in Auto-Regressive Language Models by Geva, Bastings, Filippova, Globerson (EMNLP 2023)](https://arxiv.org/abs/2304.14767)
  Traces how a subject-relation query reaches the answer: early MLPs enrich the subject position, relation then subject information flows to the last position, and attention heads extract the attribute. Studied on GPT-2 and GPT-J. Use for: lesson 0002, which part of a block does what.
- [Paper: Locating and Editing Factual Associations in GPT by Meng, Bau, Andonian, Belinkov (NeurIPS 2022)](https://arxiv.org/abs/2202.05262)
  Causal tracing that finds factual recall in mid-layer MLPs at the subject's tokens, plus ROME editing. Use for: going further on where facts live.
- [Paper: Attention Is All You Need by Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762)
  Defines scaled dot-product and multi-head attention and the residual block structure. Use for: Section 3.2 when writing attention and the causal mask.
- [Paper: RoFormer: Enhanced Transformer with Rotary Position Embedding by Su et al. (2021)](https://arxiv.org/abs/2104.09864)
  Derives rotary position embeddings as pairwise rotations of q and k. Use for: stage 1 RoPE, and again in stage 3 when cached keys must keep their original positions.
- [Paper: Root Mean Square Layer Normalization by Biao Zhang and Rico Sennrich (2019)](https://arxiv.org/abs/1910.07467)
  LayerNorm without mean-centering or bias. Use for: stage 1 RMSNorm, and as the math spec for the fused Triton RMSNorm in stage 8.
- [Paper: GLU Variants Improve Transformer by Noam Shazeer (2020)](https://arxiv.org/abs/2002.05202)
  Two pages defining SwiGLU and the other gated FFN variants. Use for: the MLP block (gate, up, down projections).
- [Paper: GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints by Ainslie et al. (2023)](https://arxiv.org/abs/2305.13245)
  Groups query heads to share K/V heads, between MHA and MQA. Use for: stage 1 head broadcasting, and stage 3 where 8 KV heads instead of 16 halves the cache.

### Sampling (Stage 2)
- [Paper: The Curious Case of Neural Text Degeneration by Holtzman et al. (2019)](https://arxiv.org/abs/1904.09751)
  Shows why greedy and beam search degenerate and introduces nucleus (top-p) sampling, comparing it against temperature and top-k. Use for: stage 2 sampler design and for understanding why the model card warns about endless repetition.

### KV cache, roofline, memory-bound decode (Stage 3)
- [Book chapter: All About Transformer Inference (How To Scale Your Model, Part 7) by Austin et al., Google DeepMind (2025)](https://jax-ml.github.io/scaling-book/inference/)
  Derives prefill vs decode arithmetic intensity, KV cache size, why decode is memory-bound at small batch, and how batch size moves you along the roofline. TPU numbers, but the algebra is hardware-agnostic. Use for: stage 3 predictions before you measure, and stage 4 to explain why batching helps.
- [Blog: Making Deep Learning Go Brrrr From First Principles by Horace He (2022)](https://horace.io/brrr_intro.html)
  Splits runtime into compute, memory bandwidth, and overhead, with operator fusion as the fix for bandwidth-bound chains. Use for: stage 3 when a 0.6B model at batch 1 turns out Python-overhead-bound rather than bandwidth-bound; stage 8 for why fusion pays.
- [Blog: Accelerating Generative AI with PyTorch II: GPT, Fast by the PyTorch team (2023)](https://pytorch.org/blog/accelerating-generative-ai-2/)
  Walks gpt-fast (now `meta-pytorch/gpt-fast`) from baseline to static KV cache plus `torch.compile`, then int8 and int4 weight-only quantization, then speculative decoding, measuring memory bandwidth utilization at each step. Use for: stages 3, 6 and 7; it is the closest existing match to this course's arc in pure PyTorch.

### Batching and paged KV cache (Stages 4 and 5)
- [Paper: Orca: A Distributed Serving System for Transformer-Based Generative Models by Yu et al. (OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu)
  Introduces iteration-level scheduling (continuous batching) and selective batching of non-attention ops. Use for: stage 4, the scheduler loop that admits and retires requests per step.
- [Paper: Efficient Memory Management for Large Language Model Serving with PagedAttention by Kwon et al. (SOSP 2023)](https://arxiv.org/abs/2309.06180)
  The vLLM paper: KV cache fragmentation analysis, block tables, copy-on-write sharing, preemption. Use for: stage 5 block allocator and block-table attention.
- [Code: Nano-vLLM by GeeeekExplorer (2025)](https://github.com/GeeeekExplorer/nano-vllm)
  About 1,200 lines of Python implementing a vLLM-style engine with paged KV cache, prefix caching, CUDA graphs and a scheduler; its benchmark uses Qwen3-0.6B on an RTX 4070 Laptop (8 GB). Use for: stages 4 and 5 as the reference to compare against after you build yours; same model, nearly the same hardware class.

### Speculative decoding (Stage 6)
- [Paper: Fast Inference from Transformers via Speculative Decoding by Yaniv Leviathan, Matan Kalman, Yossi Matias (2022)](https://arxiv.org/abs/2211.17192)
  The draft-then-verify algorithm with the modified rejection sampling that keeps the target distribution exact, plus expected-speedup analysis as a function of acceptance rate. Use for: stage 6 algorithm and the proof you should be able to reproduce.

### Quantization (Stage 7)
- [Paper: GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers by Frantar, Ashkboos, Hoefler, Alistarh (2022)](https://arxiv.org/abs/2210.17323)
  One-shot weight-only quantization to 3 to 4 bits using second-order error compensation, column by column. Use for: stage 7 after round-to-nearest int4 loses too much accuracy.
- [Paper: AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration by Lin et al. (MLSys 2024)](https://arxiv.org/abs/2306.00978)
  Protects salient weight channels by per-channel scaling chosen from activation statistics; no backprop or reconstruction. Use for: stage 7 as the simpler alternative to GPTQ, and for its discussion of on-device int4 kernels.

### Triton kernels (Stage 8)
- [Paper: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness by Dao et al. (2022)](https://arxiv.org/abs/2205.14135)
  Tiled attention with online softmax that never materializes the N by N score matrix, with an IO-complexity analysis. Use for: stage 8 algorithm before writing the kernel.
- [Docs: Triton Tutorials by the Triton project (OpenAI)](https://triton-lang.org/main/getting-started/tutorials/index.html)
  Official kernel gallery; work in order through Fused Softmax, Matrix Multiplication, Layer Normalization (adapt to RMSNorm) and Fused Attention (an FA2 implementation). Use for: stage 8, and the matmul tutorial as the base for a stage 7 int4 dequant-matmul.

### Under the tensor library (side reading for stages 1, 3 and 8)
The engine in this course sits on top of PyTorch. These cover the layer below it, where `q @ k.T` becomes a GPU kernel. None is on the spine.
- [Source: tinygrad/llm/model.py by the tinygrad authors (2026, pinned at 8e6fcc5)](https://github.com/tinygrad/tinygrad/blob/8e6fcc541f87a19711407b80dbdba8019cb82aa9/tinygrad/llm/model.py)
  A 966-line multi-architecture LLM runtime. `TransformerBlock._attention` (line 364) is Qwen3's attention in 37 lines: bias-free Q/K/V, QK-Norm, RoPE, a static preallocated fp16 KV cache written at `start_pos`, then GQA scaled dot-product attention. `Transformer` (line 664) JIT-compiles prefill and decode separately and samples with the Gumbel-max trick. `cli.py` in the same folder maps `qwen3:0.6b` to the official Q8_0 GGUF. Use for: a third reference after stage 1 and again after stage 3, to compare a static cache against yours. The file also carries MoE, MLA, state-space and AMD-only kernel paths, so read those two classes and skip the rest.
- [Source: tinygrad/llm/serve.py by the tinygrad authors (2026, pinned at 8e6fcc5)](https://github.com/tinygrad/tinygrad/blob/8e6fcc541f87a19711407b80dbdba8019cb82aa9/tinygrad/llm/serve.py)
  The server takes requests off a queue and decodes them one at a time. Line 206 carries `# TODO: batch decodes across requests (batched_step)`. Use for: stage 4, as a real engine that still lacks the scheduler you will have built; the model already accepts one `start_pos` per batch row.
- [Docs: speed in tinygrad by the tinygrad authors](https://github.com/tinygrad/tinygrad/blob/master/docs/developer/speed.md)
  Two pages splitting speed into compile, execution, scheduling and kernel speed. The Memory section defines arithmetic intensity, puts batch-1 GEMV at intensity 1, and works the 4090 numbers (1 TB/s, about 160 TFLOPS, so each loaded value must be used about 100 times). Use for: stage 3, as a second statement of the roofline argument in GPU terms.
- [Docs: tinygrad environment variables by the tinygrad authors](https://docs.tinygrad.org/env_vars/)
  `DEBUG=2` prints time, memory and bandwidth per kernel; `DEBUG=4` prints the generated kernel source. Use for: stage 8, to read the fused kernel a compiler emits for RMSNorm or attention before and after writing yours in Triton.
- [Syllabus: 10-411 Compilers for Machine Learning by George Hotz, Carnegie Mellon University (Spring 2027)](https://10-411.com/)
  A draft syllabus for an in-person course that builds an ML compiler from elementwise ops to GPU codegen, with flash attention in week 9 and SOTA inference in week 11. Lectures are not posted online and the course repository is still TBD, so today this is a topic list, not material. The [technical syllabus gist](https://gist.github.com/geohot/4768597d9dc536446ee2d5de1f29e89d) lists the ops introduced each week and says the week 5 compiler runs Qwen. Use for: a map of what lies under stage 8, and a candidate sequel once the Triton kernel is done.

### Course design
- [Paper: When Problem Solving Followed by Instruction Works: Evidence for Productive Failure by Tanmay Sinha and Manu Kapur (Review of Educational Research, 2021)](https://doi.org/10.3102/00346543211019105)
  Meta-analysis of 53 studies: problem solving before instruction beats instruction first (Hedges' g 0.36, higher with full productive-failure fidelity); it also reports where the effect reverses. Use for: justifying "attempt the stage before reading the reference implementation", and for the conditions under which that fails.

## Wisdom (Communities)
- [GPU MODE Discord](https://discord.gg/gpumode)
  Open-source GPU programming community and reading group; invite linked from the official `gpu-mode` GitHub org. Use for: Triton and kernel questions in stage 8, and profiling help in stage 3.
- [GPU MODE lectures by Mark Saroufim, Andreas Köpf, et al.](https://github.com/gpu-mode/lectures)
  Slides and notebooks for the lecture series, with recordings on the [GPU MODE YouTube channel](https://www.youtube.com/@GPUMODE). Use for: stage 8 background on CUDA and Triton, starting from lecture 1 on profiling kernels from PyTorch.
- [llama.cpp GitHub Discussions](https://github.com/ggml-org/llama.cpp/discussions)
  Active maintainer-led forum for single-machine and consumer-GPU inference. Use for: stage 7 quantization tradeoffs and stage 3 performance numbers on small GPUs.
- [vLLM Forums](https://discuss.vllm.ai/)
  Official vLLM forum with categories for scheduling, KV cache, speculative decoding and quantization; the vLLM GitHub Discussions page now pins a notice redirecting here. Use for: stages 4 to 6 design questions.

## Gaps
- No official memory bandwidth for the RTX 4050 Laptop. [NVIDIA's comparison page](https://www.nvidia.com/en-us/geforce/laptops/compare/) gives 6 GB GDDR6, 96-bit, CUDA capability 8.9, but no GB/s, and third-party spec sites disagree (192 vs 216 GB/s). Stage 3 needs this as the roofline ceiling: measure it with a large device-to-device copy benchmark instead.
- No first-party guide to Triton or `torch.compile` under WSL2. [NVIDIA's CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) covers driver and toolkit setup and notes `nvidia-smi` is limited under WSL; profiler support for kernel-level work needs to be checked on the machine before stage 8.
- No primary tutorial for a Triton weight-only int4 dequant-matmul. GPTQ and AWQ describe the math; the Triton matmul tutorial is the nearest starting point for stage 7 kernels.
- No resource on choosing a draft/target pair that fits in 6 GB. Table 1 of the Qwen3 report shows Qwen3-1.7B shares 0.6B's layer count, head layout and tokenizer, which makes 0.6B drafting for 1.7B the natural stage 6 setup, but nobody publishes acceptance rates for that pair.
- tinygrad has not been run on the laptop. Its `CUDA` runtime needs nvrtc, and whether that loads under WSL2 on the RTX 4050 is unchecked. The `NV` runtime drives the GPU directly and is unlikely to work through WSL's virtualized driver. Check with `DEV=CUDA DEBUG=2` before any lesson depends on it.
- No resource ties continuous batching to a single small GPU. Orca targets distributed serving; Nano-vLLM's code is the practical bridge.

## Unverified
- None. Every listed URL resolved and matched its title and authors. Candidates dropped for scope, not verification:
  - Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (arxiv.org/abs/2302.01318): verified, same algorithm as Leviathan et al.
  - LLM.int8() (arxiv.org/abs/2208.07339): verified, but it is mixed-precision activation and weight int8, not the weight-only scheme stage 7 builds; gpt-fast covers weight-only int8.
  - FlashAttention-2 (arxiv.org/abs/2307.08691): verified; the Triton Fused Attention tutorial implements it.
  - Kipply, "Transformer Inference Arithmetic" (kipp.ly/transformer-inference-arithmetic/): verified, overlaps the JAX inference chapter.
  - vLLM blog post (blog.vllm.ai/2023/06/20/vllm.html) and docs home (docs.vllm.ai): verified, secondary to the PagedAttention paper.
  - Karpathy "Let's build GPT" (youtube.com/watch?v=kCc8FmEb1nY), nanoGPT (github.com/karpathy/nanoGPT, README now marks it deprecated in favor of nanochat), 3Blue1Brown attention chapter (3blue1brown.com/lessons/attention): verified, prerequisite-level rather than primary.
  - vLLM GitHub Discussions: retired by the maintainers in favor of the forum above.
