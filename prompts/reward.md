# Reward function generation

You are designing a dense reward function for a Gymnasium environment.

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
def reward(obs, action, next_obs):
    ...
    return r  # float
```

## Constraints

- `obs` and `next_obs` are numpy arrays (or scalars for `Discrete` spaces); `action` is a numpy array (Box) or int (Discrete).
- Use only `numpy` (imported as `np`) and `math`. Do NOT include any import statements; `np` and `math` are already in scope.
- Return a single Python float.
- Keep the function under 60 lines.

Output ONLY the function definition. No markdown fences, no commentary.
