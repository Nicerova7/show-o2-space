# Counting: one to two

50 contrastive QA pairs in `build_pairs.json`, including objects, ideas, compliments, and poems.

- Positive ends with `Answer: two`; negative ends with `Answer: one`.
- Extract with `alignment="none"` and `tokens="last"`.
- Average `h_two - h_one` across pairs, separately per layer.
