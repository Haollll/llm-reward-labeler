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
- Name each component with a clear, descriptive key that reflects the behavioural axis it measures (e.g. a term rewarding task progress, a penalty on undesirable behaviour, a bonus for meeting a sub-goal — choose names appropriate to *this* environment and task).
- Keep the function under 60 lines.

## Example structure

This is a shape-only example. Replace `progress`, `action_cost`, and `extra_term` with components that make sense for the environment and task above, and adjust the computation accordingly.

```python
def reward(obs, action, next_obs):
    progress    =  float(...)                # a positive term capturing task progress
    action_cost = -0.1 * float(np.sum(action ** 2))  # a penalty on effort, if applicable
    extra_term  =  float(...)                # any additional shaping the task implies
    total       = progress + action_cost + extra_term
    return {{"total": total, "progress": progress, "action_cost": action_cost, "extra_term": extra_term}}
```

Output ONLY the function definition. No markdown fences, no commentary.
