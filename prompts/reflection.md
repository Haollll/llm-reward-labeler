# Reward reflection

We trained an RL policy using the reward function and trajectory summarizer
below, and tracked the values of the individual reward components in the reward
program, as well as global policy metrics (success rate, episode length,
episodic environment return), across rounds of training. Use this feedback to
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

After every round of policy training we ran a fixed number of evaluation
episodes and recorded, per round, the mean across episodes of each reward
component (summed over a trajectory), of the global policy metrics, and of the
per-episode environment return. Each list below has one entry per round, most
recent last. `Max`, `Mean`, `Min` are computed over the full list.

{training_summary}

## How to analyze the policy feedback

Carefully analyse each existing reward component and global metric in the
manner suggested below, then write the new reward function and the new
trajectory summarizer:

1. If `success_rate` (when reported) or `episode_env_reward` stays near zero
   across rounds, the current reward function is failing to drive task
   progress at all. You must rewrite the entire reward function from scratch.
2. If a reward component's values are near-identical across rounds (compare
   the list against `Min` / `Max`), then RL is not able to optimize this
   component as it is written. You may consider:
   - (a) Changing its scale or the value of its temperature parameter
   - (b) Re-writing the reward component
   - (c) Discarding the reward component
3. If some reward component's magnitude is significantly larger than the
   others, you must rescale its value to a comparable range so it does not
   drown the other components out.
4. If `episode_length` is much shorter than the environment's maximum episode
   length and is not improving, the policy is terminating early. Investigate
   which component(s) drive premature termination and re-balance.

Analyse each existing reward component using the framework above *first*, then
write the new reward function code.

## How to update the trajectory summarizer

The trajectory summarizer turns a trajectory into a *string* read by a
comparator LLM that labels pairwise preferences over trajectories. If you
change which components exist in the reward function, the summarizer must
change in lockstep so the comparator can still see the new components:

- For every named component in the new reward function, the summarizer must
  report per-step mean and trajectory sum.
- The summarizer must **not** include the total reward (the comparator anchors
  on totals when shown them — keep total out of the string).
- The summarizer should also surface behavioural features inferred from the
  `obs`, `next_obs`, and `action` streams (means, stds, sums, deltas) so the
  comparator can reason about behaviour, not just per-component scalars.

## Tips for writing the reward function

- You may find it helpful to normalize a reward component to a fixed range by
  applying transformations like `np.exp(-x / temp)`. If you do, you must
  introduce a temperature parameter as a named local variable inside the
  function body (not an input). Each transformed component should have its
  own temperature variable.
- Numerical safety: every value you return must be finite (no `nan`, no
  `inf`). Pre-clip the input to `np.exp` (e.g. `np.exp(-np.clip(x / temp, -20.0, 20.0))`),
  guard divisions against zero denominators, and avoid `np.log` on
  potentially non-positive values. A reward function that returns a single
  non-finite value will be rejected and your changes discarded.
- The function signature is `def reward(obs, action, next_obs) -> dict` and
  the returned dict must contain a `"total"` key (sum of the other entries)
  plus one key per named component. Component names should describe the
  behavioural axis they measure (clear, env-appropriate names).
- `np` and `math` are in scope; do NOT include any import statements.

## Tips for writing the trajectory summarizer

- The function signature is `def summarize(trajectory) -> str`.
- `trajectory` is a list of `(obs, action, next_obs, r_comp, done)` tuples
  where `r_comp` is the dict produced by the reward function. Iterate
  generically over `r_comp.keys()` and skip the `"total"` key entirely.
- Use the same metrics in the same order every call so two summaries are
  directly comparable.
- `np` and `math` are in scope; do NOT include any import statements.

## Output

Respond with a single JSON object — no markdown fences, no extra keys.
Either `reward_code` or `semantic_code` may be `null` to leave that artifact
unchanged, but at least one must be a complete function. Both fields, when
non-null, must be complete Python source as a string.

{{"analysis": "<paragraph walking through each reward component (flat? dominating? changing?) and each global metric (success rate trend, episode length trend, episode_env_reward trend)>", "reward_code": "<complete def reward(obs, action, next_obs) -> dict source, OR null to keep current>", "semantic_code": "<complete def summarize(trajectory) -> str source, OR null to keep current>", "reasoning": "<why these specific changes follow from the training feedback above>"}}
