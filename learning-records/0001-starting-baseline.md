# Starting baseline: shapes yes, transformer internals no

Self-reported at the start (2026-11-09): can do matrix multiplication by hand and predict shapes like `(B, T, D) @ (D, V)`. Has not worked with softmax/cross-entropy numerics, has not trained a network in PyTorch, has not read the transformer paper or seen Karpathy's GPT video, and could not define a KV cache. Did CUDA parallel programming coursework but considers it mostly forgotten.

**Implications**: Lesson 1 teaches softmax and its overflow from scratch. PyTorch tensor idioms (indexing, broadcasting, `keepdim`) need short in-lesson notes until the learner shows fluency. GPU programming in stage 8 starts from the execution model again, not from a refresher.
