# Trajectory preference comparison

You are comparing two full-episode trajectories from a Gymnasium environment to
express a preference used as a Bradley-Terry training label.

**Task description:** {task}

## Trajectory A

{summary_a}

## Trajectory B

{summary_b}

## What you are deciding

Answer one question:

> Which trajectory — A or B — achieves the **task** above more successfully?

Task achievement is a judgement about behaviour, not a comparison of scalar
totals. The summaries **deliberately omit total reward** so the decision cannot
collapse to "whichever number is bigger". Each summary gives you:

1. **Behavioural features (decisive).** From the task description, decide what
   "doing the task well" looks like, then judge which trajectory's behaviour is
   closer to that, using the concrete behavioural statistics in each summary.
2. **Per-component reward breakdown.** Compare component-by-component; for each
   named component decide which trajectory did better and whether that axis is
   task-relevant. Winning on the components that matter to the task is what
   counts.

### Hard rule

The summaries contain no total reward and you must not reconstruct one (e.g. by
summing per-component values). A justification that invents or references a
scalar "total" / "overall" / "sum" is **not acceptable**. Reason about behaviour
and individual components only.

### Reasoning structure (work through before answering)

1. **Task achievement (A vs B).** What does success look like, and which
   trajectory's behavioural features are closer to it — on which features?
2. **Component-by-component (A vs B).** For each component, which trajectory was
   better, and is that component task-relevant?
3. **Tally.** Which trajectory wins on the *task-relevant* features and
   components? That is your preference.

Your justification must name at least one **specific behavioural feature** and
cite at least one **specific reward component by name**, and must NOT reference a
total / overall / summed reward.

## Required output

Output a single JSON object with exactly two keys:

```json
{{"preference": "A" or "B", "explanation": "<one to three sentences citing (a) a behavioural feature on which the preferred trajectory better achieves the task, and (b) at least one specific reward component by name; do NOT reference a 'total'/'overall'/summed reward>"}}
```

No markdown fences, no extra commentary — just the JSON object.
