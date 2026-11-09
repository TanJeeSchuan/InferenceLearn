# Built embed, RMSNorm, LM head and stable softmax

The learner passed all 9 lesson 0001 tests (2026-11-09) with their own `embed`, `rms_norm`, `lm_head`, `softmax` and `zero_layer_logits`. On the way they needed three corrections: square before mean in RMSNorm, sum the exponentials in softmax rather than the raw logits, and return logits from the model rather than probabilities. Each fix took one round once the slip was pointed out.

**Evidence**: the pytest run on the laptop, and code pasted in the session that used named intermediates after LR 0003's suggestion.

**Implications**: Lesson 0002 can assume these four ops work and can be reused. The learner understands that the model returns logits and that softmax is a separate, later step, which stage 2 sampling depends on. Keep watching for reductions applied to the wrong tensor (LR 0003) in attention.
