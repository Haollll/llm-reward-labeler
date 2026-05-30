# LLM Reward Labeler

A system that trains RL agents without access to ground-truth rewards by using LLMs as drop-in replacements for human annotators in RLHF-style reward learning.

---

## Project Overview

Standard RLHF requires human annotators to compare trajectory pairs and label preferences. This project replaces that human-in-the-loop with an LLM pipeline:

1. **LLM-generated reward function** — given the environment source code, an LLM writes a fixed reward function `r_fixed` capturing the task objective
2. **LLM-generated semantic layer** — an LLM writes a function that translates raw trajectories (states, actions) into natural language descriptions
3. **LLM preference labeler** — an LLM compares pairs of trajectory descriptions and returns a preference label, replacing human annotators
4. **Bradley-Terry ensemble reward model** — a learned reward model `R_phi` (ensemble of 3 networks) is trained on these LLM-generated labels
5. **Combined reward signal** — `r_total = alpha * r_fixed + (1 - alpha) * g * normalise(R_phi)`, where `alpha` decays across training rounds
6. **Active learning** — candidate trajectories are ranked by ensemble disagreement; only the most informative pairs are queried, maximizing label efficiency

The agent is PPO on `HalfCheetah-v5`. An `--oracle` flag replaces the LLM labeler with ground-truth reward sums for ablation comparison.

---

## File Structure

```
.
├── train.py              # Entry point, argparse
├── evaluate.py           # Load saved policy/reward model, render, plot rewards
├── trainer.py            # Trainer, TrainerConfig, main training loop
├── ppo_agent.py          # PPOAgent wrapping stable-baselines3 PPO
├── sampler.py            # Trajectory collection, oracle/LLM labelling, active learning
├── reward_model.py       # EnsembleRewardModel, PreferenceBuffer, traj_to_segment
├── env_setup.py          # CustomRewardWrapper, collect_trajectory
├── llm_utils.py          # LLM code generation (reward fn, semantic layer, comparison), caching
├── helper.py             # Task loading, pretty printing
├── prompts/
│   ├── reward.md         # Prompt template: LLM-generated reward function
│   ├── semantic.md       # Prompt template: trajectory-to-text semantic layer
│   └── compare.md        # Prompt template: pairwise trajectory comparison
└── tasks/
    ├── halfcheetah.txt         # Task description with full obs layout (recommended)
    └── halfcheetah_minimal.txt # Minimal task description without obs layout (tests reflection)
```

---

## Installation

```bash
# conda
conda create -n llm-reward python=3.11
conda activate llm-reward
pip install gymnasium[mujoco] torch stable-baselines3 openai python-dotenv

# or pip only
pip install gymnasium[mujoco] torch stable-baselines3 openai python-dotenv
```

Set your OpenAI key in a `.env` file:

```
OPENAI_API_KEY=sk-...
```

---

## Usage

```bash
# Quick test with oracle labels (no LLM queries needed)
python train.py --oracle --rounds 3 --queries 5 --ppo-steps 5000

# Recommended oracle run (tuned reward model settings)
python train.py --oracle --rounds 9 --queries 10 --ppo-steps 20000 \
    --reward-epochs 50 --lambda-smooth 0.05 --dynamic-batch

# Full run with LLM as preference labeler
python train.py --rounds 9 --queries 10 --ppo-steps 50000 \
    --reward-epochs 50 --lambda-smooth 0.05 --dynamic-batch

# Evaluate saved artifacts, render the scene, and save reward trace plot
python evaluate.py

# Test env setup + LLM code generation
python env_setup.py
```

| Argument | Default | Description |
|---|---|---|
| `--rounds` | 9 | Number of active learning rounds |
| `--queries` | 10 | Preference queries per round |
| `--ppo-steps` | 50000 | PPO environment steps per round |
| `--reward-epochs` | 50 | Reward model training epochs per round |
| `--lambda-smooth` | 0.05 | Temporal smoothness penalty coefficient for reward model |
| `--dynamic-batch` | off | Scale batch size with buffer: `max(8, min(32, buffer//2))` instead of fixed 10 |
| `--oracle` | off | Use ground-truth reward sums instead of LLM labels |

---

## Key Design Decisions

- **LLM as preference labeler** — replaces human annotators; the LLM sees natural-language trajectory descriptions and picks the preferred one
- **Ensemble reward model (×3)** — three independent networks trained on the same preference buffer; ensemble variance estimates label uncertainty
- **Active learning** — collect 5× candidate trajectory pairs per round, score each by ensemble disagreement, query only the top-`n` by disagreement (cold start round uses uniform sampling)
- **`CustomRewardWrapper`** — backward-compatible; `reward_model=None` uses only `r_fixed`, making it easy to ablate the learned component
- **LLM output caching** — generated reward and semantic functions are cached to disk, avoiding redundant API calls across runs
- **`lambda_smooth=0.05`** — the temporal smoothness penalty is kept small intentionally; see the debug note below for why a larger value causes the reward model to stall
- **Explicit obs layout in task description** — `tasks/halfcheetah.txt` annotates every observation dimension (e.g. "obs[8] is forward velocity, not position"). Without this, the LLM can misinterpret the observation semantics and generate a reward function anti-correlated with the true objective; see the debug note below

---

## Reward Formulation

```
r_total = alpha * r_fixed(s, a, s') + (1 - alpha) * g * normalise(R_phi(s, a))
```

- `r_fixed` — LLM-generated fixed reward function derived from env source code
- `R_phi` — ensemble reward model trained on LLM preference labels
- `alpha` — linearly decays from 1.0 across training rounds, shifting from fixed reward toward learned reward
- `g` — scalar gain controlling the learned reward's contribution
- `normalise` — running z-score normalization of `R_phi` outputs

---

## Ablation Experiments

The `--oracle` flag replaces the LLM labeler with the ground-truth environment reward sum for each trajectory segment. This provides an upper-bound comparison: oracle labeling shows how well the pipeline works when preference labels are perfect, isolating errors introduced by LLM annotation noise.

---

## Reward Model Debug Notes

**Symptom:** reward model loss stuck at ~2.0, accuracy at 60–70%.

**Root cause:** `lambda_smooth=1.0` (original default) was too large. At that scale the temporal smoothness penalty grows at roughly the same rate as the CE loss improves, so the two cancel out and the reported total loss appears flat. The model was actually learning — CE was declining — but the smooth term masked it entirely.

Mechanically: at random initialization, network outputs are near zero so smooth ≈ 0. As the network starts making non-trivial reward predictions, smooth rises. With `lambda_smooth=1.0` and three ensemble members:

```
total loss ≈ 3 × (CE_m − Δ) + 1.0 × 3 × (smooth_m + Δ) ≈ 3 × ln(2) ≈ 2.08   (stuck)
```

With `lambda_smooth=0.05` the smooth contribution is small enough that CE improvements show through:

```
total loss = CE_total + 0.05 × smooth_total
           ≈ 1.49    + 0.05 × 2.82          ≈ 1.63   (and declining)
```

**Fix:** set `--lambda-smooth 0.05`. Results on oracle mode, 5 rounds:

| Round | CE loss | smooth (unscaled) | total | acc |
|---|---|---|---|---|
| cold start | 1.691 | 1.603 | 1.771 | 100% |
| 1 | 1.537 | 2.711 | 1.672 | 100% |
| 2 | 1.522 | 3.032 | 1.674 | 94% |
| 3 | 1.521 | 2.836 | 1.663 | 98% |
| 4 | 1.502 | 2.784 | 1.642 | 97% |
| 5 | 1.489 | 2.820 | 1.630 | 97% |

Env reward improved from −38 → +108 over the same 5 rounds.

**Related code changes (all in this fix):**
- `TrainerConfig`: added `lambda_smooth` (default `0.05`) and `dynamic_batch` fields
- `reward_model.py`: `train()` now returns `(ce_loss, smooth_loss)` separately so both can be monitored
- `trainer.py`: logs CE and smooth independently; fixed a bug where `label_accuracies.append(acc)` was never called
- `train.py`: exposed `--lambda-smooth` and `--dynamic-batch` CLI flags

---

## LLM Reward Function Debug Notes

**Symptom:** env reward collapsed as `alpha` decreased — the more the agent relied on `R_phi`, the worse it performed. In oracle mode, 9 rounds:

| Round | alpha | env reward |
|---|---|---|
| 2 | 0.875 | +361 |
| 4 | 0.625 | −57 |
| 6 | 0.375 | −112 |
| 8 | 0.125 | −328 |

Confusingly, `R_phi`'s CE loss was *declining* (1.45 → 1.13) and accuracy held at 90%+ — the reward model looked healthy by its own metrics.

**Diagnosis:** `diagnose_reward_model.py` measured the Pearson correlation between `R_phi`'s per-step predictions and the true environment reward over 5000 steps:

```
Step-level    Pearson r : −0.17
Segment-level Pearson r : −0.49
```

Both correlations are **negative**: `R_phi` was assigning *high* predicted rewards to states that produce *low* true env reward. PPO maximising `R_phi` therefore drove env reward down. The segment-level magnitude (0.49) being much larger than the step-level (0.17) also confirmed that R_phi only carries meaningful signal at segment granularity — consistent with how it was trained.

**Root cause:** the LLM-generated `reward_fn` misread the HalfCheetah observation layout. `obs[8]` is already **forward velocity** (as defined by MuJoCo), but the LLM treated it as a position and computed `dx = next_obs[8] − obs[8]`, effectively rewarding *acceleration* rather than velocity. This made `r_fixed` (and the oracle labels derived from it) anti-correlated with the true objective, so `R_phi` learned the wrong ranking.

**Fix:** `tasks/halfcheetah.txt` was updated to include an explicit per-dimension obs layout, clearly labelling `obs[8]` as velocity. With the corrected task description, the LLM generates a reward function that directly reads velocity, eliminating the sign inversion.

**Results after fix** (oracle mode, 9 rounds):

| Round | alpha | env reward |
|---|---|---|
| 2 | 0.875 | +294 |
| 4 | 0.625 | +453 |
| 6 | 0.375 | +494 |
| 8 | 0.125 | +556 |

Env reward now *increases* as alpha decreases — the agent learns to rely on `R_phi` and improves.

**Broader insight:** LLMs routinely misinterpret observation vectors for physics simulators because the variable names in MuJoCo XML are terse and the semantics (position vs velocity vs angle) are not obvious from the name alone. Providing an annotated obs layout in the task description is a simple, high-leverage fix. An alternative is to rely on the reflection mechanism to detect and correct misaligned reward functions at runtime — `tasks/halfcheetah_minimal.txt` omits the obs layout intentionally to test whether reflection can recover from this class of error.

---

## References

- Christiano et al. (2017). [Deep Reinforcement Learning from Human Preferences.](https://arxiv.org/abs/1706.03741) NeurIPS 2017.
- Lee et al. (2021). PEBBLE: Feedback-Efficient Interactive Reinforcement Learning via Relabeling Experience and Unlabeled Data. ICML 2021. *(ensemble reward model design)*
- Ma et al. (2023). EUREKA: Human-Level Reward Design via Coding Large Language Models. *(LLM-as-reward-designer inspiration)*
