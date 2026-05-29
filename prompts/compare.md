# Trajectory preference comparison

You are comparing two trajectories from a Gymnasium environment to express a
preference for Bradley-Terry preference learning.

**Task description:** {task}

## Trajectory A

{summary_a}

## Trajectory B

{summary_b}

## What you are actually deciding

You are answering one question:

> Which trajectory — A or B — achieves the **task** above more successfully?

Task achievement is a judgement about behaviour, not a comparison of scalar
totals. The summaries you are given **deliberately omit total reward** so that
this judgement cannot collapse to "whichever number is bigger". Each summary
contains two kinds of information:

1. **Behavioural features (decisive).** Reading the task description, decide
   what "doing the task well" actually looks like. Then ask: which trajectory's
   *behaviour* more closely matches that? Use the concrete behavioural features
   in each summary (observation-derived stats, action stats, motion patterns,
   progress markers, failure cues) to answer this.
2. **Per-component reward breakdown.** Compare the two trajectories
   **component-by-component**. For each named component, decide which
   trajectory did better on that axis and whether that axis is task-relevant.
   A trajectory winning on the components that matter to the task is preferable,
   even if it loses on components that are less task-aligned.

### Hard rule

The summaries do not contain a total reward and you must not attempt to
reconstruct one (e.g. by summing the per-component values you see). A
justification that invents or references a scalar "total" / "overall reward"
is **not acceptable**. Reason about behaviour and individual components only.

### Required reasoning structure

Work through these in order before stating your preference:

1. **Task achievement (A vs B).** From the task description, what does success
   look like? Which trajectory's behavioural features are closer to that — and
   on which specific features?
2. **Component-by-component comparison (A vs B).** For each reward component,
   which trajectory scored better, and is that component task-relevant?
3. **Tally.** Does A or B win on the *task-relevant* components and
   behavioural features? That is your preference.

Your justification must:
- name at least one **specific behavioural feature** showing task achievement, AND
- cite at least one **specific reward component** (by name) on which the
  preferred trajectory either won or lost.
- *not* reference a "total" / "overall" / "sum" of reward.

## Required output

Output a single JSON object with exactly two keys:

```json
{{"preference": "A" or "B", "explanation": "<one to three sentences citing (a) a behavioural feature on which the preferred trajectory better achieves the task, and (b) at least one specific reward component by name; do NOT reference a 'total' / 'overall' / summed reward>"}}
```

No markdown fences, no extra commentary — just the JSON object.
