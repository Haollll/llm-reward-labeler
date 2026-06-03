# Reward reflection (Phase I)

We are in **Phase I** of training: an RL policy is optimized purely against the
coded reward function below, and a trajectory summarizer feeds an LLM preference
labeler that trains a Bradley-Terry reward model. After each round we track the
individual reward components and global policy metrics. Use this feedback to
produce an improved reward function AND an improved trajectory summarizer.

## Task

{task}

## Current reward function

```python
{reward_code}
```

## Current trajectory summarizer

```python
{semantic_code}
```

## Training feedback

Each list below has one entry per Phase-I round, most recent last. `Max`, `Mean`,
`Min` are computed over the full list.

{training_summary}

## How to analyze the feedback

Analyse each reward component and global metric, then rewrite both functions:

1. If `episode_env_reward` stays near zero or does not improve across rounds, the
   reward function is not driving task progress — rewrite it substantially.
2. If a component's values are near-identical across rounds (compare against its
   `Min`/`Max`), RL cannot optimize it as written. Consider rescaling it,
   re-writing it, or discarding it.
3. If one component's magnitude dwarfs the others, rescale it to a comparable
   range so it does not drown the others out.
4. If `episode_length` is much shorter than the env's maximum and not improving,
   the policy is terminating early — find the component(s) driving premature
   termination and re-balance (e.g. add/raise a survival term).

## How to update the trajectory summarizer

The summarizer turns a trajectory into the *string* read by the comparator LLM.
If you change which components exist in the reward function, the summarizer must
change in lockstep:

- For every named component in the new reward function, report per-step mean and
  trajectory sum.
- Do **not** include the total reward (the comparator anchors on totals).
- Also surface behavioural features inferred from the `obs`, `next_obs`, and
  `action` streams so the comparator can reason about behaviour, not just
  per-component scalars.

## Constraints on the code you write

- `reward` signature: `def reward(obs, action, next_obs) -> dict` returning a
  `"total"` key (sum of the others) plus one key per named component.
- `summarize` signature: `def summarize(trajectory) -> str`; iterate generically
  over `r_comp.keys()` and skip `"total"`.
- `np` and `math` are in scope; do NOT include import statements.
- Every returned reward value must be finite — pre-clip `np.exp` arguments, guard
  divisions, avoid `np.log` on non-positive values. Non-finite output is rejected.

## Output

Respond with a single JSON object — no markdown fences, no extra keys. Either
`reward_code` or `semantic_code` may be `null` to leave that artifact unchanged,
but at least one must be a complete function.

{{"analysis": "<walk through each reward component (flat? dominating? changing?) and each global metric (episode length trend, episode_env_reward trend)>", "reward_code": "<complete def reward(obs, action, next_obs) -> dict source, OR null>", "semantic_code": "<complete def summarize(trajectory) -> str source, OR null>", "reasoning": "<why these changes follow from the feedback>"}}
