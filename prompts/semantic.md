# Semantic trajectory summarizer

You are designing a function that summarizes a trajectory from a Gymnasium environment as a short natural-language description. Focus on SOFT QUALITIES — smoothness, stability, energy use, directionality, progress toward the goal, and any failure cues — the kind of things a human evaluator would notice when comparing two trajectories.

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

`trajectory` is a list of tuples `(obs, action, next_obs, reward, done)` where `obs` and `next_obs` are numpy arrays (or scalars), `action` is a numpy array or int, `reward` is a float, and `done` is a bool.

## Constraints

- Use only `numpy` (as `np`) and `math`. Do NOT include any import statements.
- Compute meaningful aggregate stats (means, stds, sums, deltas) and reduce them to natural language.
- Return a multi-line string. Include concrete numerical values where they help comparison.
- The output is consumed by another LLM that compares two summaries to express a preference, so be concrete and comparable across trajectories — same metrics in the same order each time.

Output ONLY the function definition. No markdown fences, no commentary.
