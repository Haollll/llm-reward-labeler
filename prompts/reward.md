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
    return {{
        "total": float,        # sum of all components — used as the actual reward
        "component_name": float,  # one key per reward component
        ...
    }}  
```
## Constraints

- `obs` and `next_obs` are numpy arrays (or scalars for `Discrete` spaces); `action` is a numpy array (Box) or int (Discrete).
- Use only `numpy` (imported as `np`) and `math`. Do NOT include any import statements; `np` and `math` are already in scope.
- Return a **dict** with a `"total"` key (float) and one key per reward component.
- Name each component clearly (e.g. `"velocity"`, `"smoothness"`, `"energy"`, `"survival"`).
- Keep the function under 60 lines.

## Example structure
 
```python
def reward(obs, action, next_obs):
    velocity   =  next_obs[8]
    smoothness = -float(np.mean(np.abs(np.diff(action)))) if len(action) > 1 else 0.0
    energy     = -0.1 * float(np.sum(action ** 2))
    total      = velocity + smoothness + energy
    return {{"total": total, "velocity": velocity, "smoothness": smoothness, "energy": energy}}
```

Output ONLY the function definition. No markdown fences, no commentary.
