# Knowledge isn't in the embeddings

Before running the zero-layer model, the learner predicted it would answer "Paris" after "The capital of France is". It actually echoed the input token at every position. The misconception was that embed, norm and head alone carry the model's knowledge. In fact, with tied weights those three can only measure how similar tokens are, and every fact the model knows lives in the 28 blocks.

The learner also added a trailing space to the prompt. That turned the space into its own token, and the full model then predicted digits instead of " Paris", because BPE folds the leading space into word tokens like " Paris".

**Implications**: Lesson 0002 should open by asking where "Paris" comes from inside the blocks, and have the learner commit to a guess before teaching attention vs MLP. Tokenizer edge cases, like a trailing space, are worth a sidenote in stage 2 when prompts get built.
