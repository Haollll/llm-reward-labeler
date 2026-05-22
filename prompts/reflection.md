# Reward reflection

You are an expert RL reward designer. Given the training status below,
choose ONE targeted fix.

## Training status

{status}

## Current r_fixed

```python
{reward_code}
```

## Current summarize

```python
{semantic_code}
```

## Task

{task}

## Rules

- `acc < 0.70`       → pick **B** (semantic layer is missing key behavioural features)
- reward declining   → pick **A** (penalties are too aggressive)
- reward plateauing  → pick **A** or **C**
- reward improving   → pick **C** (raise the comparison bar)

## Options

- **A** — rewrite `r_fixed`: `def reward(obs, action, next_obs) -> float` (`np` and `math` in scope, no imports)
- **B** — rewrite `summarize`: `def summarize(trajectory) -> str` (`np` and `math` in scope, no imports; `trajectory` is a list of `(obs, action, next_obs, reward, done)`)
- **C** — write a short guidance string (1–3 sentences) for the LLM comparator

## Output

Respond with a single JSON object — no markdown fences, no extra keys:

{"diagnosis": "one sentence", "option": "A" or "B" or "C", "code": "complete Python function (A or B) or null", "fn_name": "reward" or "summarize" or null, "guidance": "short string (C) or null", "reasoning": "why this change should help"}