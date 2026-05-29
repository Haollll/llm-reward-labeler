# Semantic trajectory summarizer

You are designing a function that summarizes a trajectory from a Gymnasium
environment as a short natural-language description. Focus on the behavioural
qualities a human evaluator would notice when comparing two trajectories — the
features that distinguish a competent rollout from a poor one for *this specific
environment and task*. Infer which features matter from the environment
description and task below; do not assume any particular domain (locomotion,
manipulation, control, etc.).

The output is consumed by a comparator LLM that is explicitly told **not** to
decide on total reward alone. Your summary must therefore surface enough
qualitative behavioural information that a sensible qualitative judgement is
possible. If the summary leads with total reward or makes it the most prominent
number, the comparator will fall back to picking the higher number — defeating
the purpose of preference learning. **Lead with behaviour, not total.**

**Environment ID:** {env_id}
**Observation space:** {obs_space}
**Action space:** {act_space}

**Environment docstring:**
```
{docstring}
```

**Task description:** {task}

## Required output

Write a Python function with the EXACT signature:

```python
def summarize(trajectory):
    ...
    return summary  # str
```

`trajectory` is a list of tuples `(obs, action, next_obs, r_comp, done)` where
`obs` and `next_obs` are numpy arrays (or scalars), `action` is a numpy array or
int, `r_comp` is a **dict** of reward components keyed by name (`"total"` is
always present and equals the sum of the other components; the other keys are
defined by the reward function currently in use and you can iterate over them
generically), and `done` is a bool.

## Output structure (order matters)

The summary is read by a comparator LLM that has been observed to anchor on
total reward whenever it is present in the input. To force qualitative
reasoning, the summary string MUST NOT include the total reward at all — neither
the value, nor a "total" / "overall" / "sum" line, nor any phrase that reduces
the trajectory to a single scalar. The per-component breakdown gives the
comparator everything it needs.

Build the summary string in this order:

1. **Behavioural features (most prominent).** Concrete numerical descriptions
   of the behaviour the task implies — derived from `obs`, `next_obs`, and
   `action` streams. Use means, stds, sums, deltas, extremes; report enough
   that two trajectories' behaviours are *directly comparable* on the same
   axes.
2. **Per-component reward breakdown.** Iterate generically over the keys of
   `r_comp` and **skip the `"total"` key entirely** (`for k in r_comp: if k ==
   "total": continue`). For each remaining component report: per-step mean and
   trajectory sum. This reveals *how* the reward was assembled without giving
   the comparator a scalar to anchor on.

Do NOT include any line that reports `r_comp["total"]`, `sum(r_comp.values())`,
or any equivalent scalar summary.

## Constraints

- Use only `numpy` (as `np`) and `math`. Do NOT include any import statements.
- Compute meaningful aggregate stats (means, stds, sums, deltas, extremes) and
  reduce them to natural language.
- Return a multi-line string. Include concrete numerical values where they help
  comparison.
- The output is consumed by another LLM that compares two summaries, so be
  concrete and comparable across trajectories — same metrics in the same order
  every call.

Output ONLY the function definition. No markdown fences, no commentary.
