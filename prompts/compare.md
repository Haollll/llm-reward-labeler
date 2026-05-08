# Trajectory preference comparison

You are comparing two trajectories from a Gymnasium environment to express a preference for Bradley-Terry preference learning.

**Task description:** {task}

## Trajectory A

{summary_a}

## Trajectory B

{summary_b}

## Required output

Decide which trajectory better accomplishes the task above. Consider the soft qualities described in each summary (smoothness, stability, energy use, directionality, progress) as well as overall task success.

Output a single JSON object with exactly two keys:

```json
{{"preference": "A" or "B", "explanation": "<one to three sentences justifying the choice, referencing concrete numbers from the summaries>"}}
```

No markdown fences, no extra commentary — just the JSON object.
