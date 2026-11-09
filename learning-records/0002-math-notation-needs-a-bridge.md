# Math notation needs a bridge to code

After reading lesson 0001, the learner asked for help with the notation (2026-11-09). Formulas written with Σ, ⊙, subscripts, ᵀ and diag(·) don't yet read fluently, even though matrix shapes do.

**Implications**: Every formula in a lesson comes with the PyTorch line that computes it and a small numeric example (3 to 4 elements) the learner can check by hand. New symbols go into `reference/notation.html` the first time they appear. Once the learner reads Σ/⊙/ᵀ without the decoder, drop the inline crutches; watch for this in their questions and shape traces.
