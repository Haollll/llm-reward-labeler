# Coded reward function generation

You are designing a dense, component-wise **coded reward function** for a
Gymnasium environment. This function is the `r_fixed` term that bootstraps RL
training and anchors a Bradley-Terry reward model learned from LLM preferences.

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
        "total": float,           # sum of all components — the scalar reward
        "component_name": float,  # one key per named reward component
        ...
    }}
```

## Constraints

- `obs` and `next_obs` are numpy arrays (or scalars for `Discrete` spaces);
  `action` is a numpy array (Box) or int (Discrete).
- Use only `numpy` (as `np`) and `math`. Do NOT include any import statements;
  `np` and `math` are already in scope.
- Return a **dict** with a `"total"` key (float, the sum of the other entries)
  plus one key per reward component.
- Name each component with a clear, descriptive key reflecting the behavioural
  axis it measures (task progress, an energy/effort penalty, a stability or
  survival term, etc. — choose names appropriate to *this* environment).
- Keep components on roughly comparable scales so no single term dominates.
- Numerical safety: every returned value must be finite. Pre-clip arguments to
  `np.exp` (e.g. `np.exp(-np.clip(x, -20.0, 20.0))`), guard divisions, and avoid
  `np.log` on non-positive values.
- Keep the function under 60 lines.

## Example structure (shape only — replace with task-appropriate components)

```python
def reward(obs, action, next_obs):
    progress    =  float(...)                          # task progress
    action_cost = -0.1 * float(np.sum(np.square(action)))  # effort penalty
    stability   =  float(...)                          # posture / survival term
    total       = progress + action_cost + stability
    return {{"total": total, "progress": progress,
            "action_cost": action_cost, "stability": stability}}
```

Output ONLY the function definition. No markdown fences, no commentary.
