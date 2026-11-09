# Reductions land on the wrong intermediate

In lesson 0001 the learner made the same slip twice in a row. In RMSNorm they wrote `x.mean().pow(2)` where the formula is mean(x²), and in softmax they wrote `z.exp() / z.sum()` where the formula is exp(z) / Σ exp(z). Both times the shapes were right and the reduction ran on the wrong tensor. Nested notation like f(g(x)) is being read left to right as "do f, then g", not inside out.

**Implications**: Teach the habit of naming every intermediate after the formula (`sq = x ** 2`, `ms = sq.mean(...)`, `e = z.exp()`, `total = e.sum(...)`), so each reduction has exactly one obvious input. Shape traces in lesson 0002 should list the formula's intermediates in evaluation order. Watch attention (softmax of QKᵀ/√d, then @ V) for the same slip.
