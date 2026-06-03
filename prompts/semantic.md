# Semantic trajectory summarizer generation

You are designing a function that turns one **full-episode trajectory** from a
Gymnasium environment into a short, behaviourally-focused text summary. The
summary is read by a comparator LLM that labels pairwise preferences over
trajectories to train a Bradley-Terry reward model.

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
`obs`/`next_obs` are numpy arrays (or scalars), `action` is a numpy array or int,
`r_comp` is the **dict** of reward components produced by the current reward
function (`"total"` is always present and equals the sum of the others; the other
keys vary and you must iterate over them generically), and `done` is a bool.

## Output structure (order matters)

The comparator anchors on total reward whenever it is present, which defeats
preference learning. Your summary string **MUST NOT include the total reward** —
not the value, not a "total"/"overall"/"sum" line, nor any single scalar that
collapses the trajectory. Build the summary in this order:

1. **Behavioural features (most prominent).** Concrete numerical descriptions of
   the behaviour the task implies, derived from the `obs`, `next_obs`, and
   `action` streams — means, stds, sums, deltas, extremes (e.g. average forward
   velocity, height stability, energy expended, how long the agent stayed
   "healthy", episode length). Report enough that two trajectories are *directly
   comparable on the same axes*.
2. **Per-component reward breakdown.** Iterate generically over `r_comp` and
   **skip the `"total"` key** (`for k in r_comp: if k == "total": continue`).
   For each remaining component report its per-step mean and trajectory sum.

## Constraints

- Use only `numpy` (as `np`) and `math`. Do NOT include any import statements.
- Use the **same metrics in the same order every call** so two summaries are
  directly comparable.
- Return a multi-line string with concrete numerical values.
- Handle a trajectory of any length (including very short episodes) without error.

Output ONLY the function definition. No markdown fences, no commentary.
