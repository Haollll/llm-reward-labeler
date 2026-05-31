# LLM Reward Labeler

A system that trains RL agents without access to ground-truth rewards by using LLMs as drop-in replacements for human annotators in RLHF-style reward learning, across multiple MuJoCo environments.

---

## Project Overview

Standard RLHF requires human annotators to compare trajectory pairs and label preferences. This project replaces that human-in-the-loop with an LLM pipeline:

1. **LLM-generated reward function** — given the environment source code, an LLM writes a fixed reward function `r_fixed` capturing the task objective
2. **LLM-generated semantic layer** — an LLM writes a function that translates raw trajectories (states, actions) into natural language descriptions
3. **LLM preference labeler** — an LLM compares pairs of trajectory descriptions and returns a preference label, replacing human annotators
4. **Bradley-Terry ensemble reward model** — a learned reward model `R_phi` (ensemble of 3 networks) is trained on these LLM-generated labels
5. **Combined reward signal** — `r_total = alpha * r_fixed + (1 - alpha) * normalise(R_phi)`, where `alpha` decays across training rounds
6. **Active learning** — candidate trajectories are ranked by ensemble disagreement; only the most informative pairs are queried, maximizing label efficiency
7. **Reflection** — after each evaluation round, the LLM inspects the round-over-round training dynamics and may rewrite `r_fixed` and the trajectory summarizer (EUREKA-style)

The agent is PPO (stable-baselines3). The pipeline is compared against **RL-Zoo3-trained PPO baselines** on the true environment reward.

### Supported environments

`Pendulum-v1`, `Swimmer-v4`, `HalfCheetah-v4`, `Hopper-v4`, `Walker2d-v4`, `Ant-v4` — each the most up-to-date version RL-Zoo3 ships tuned PPO hyperparameters for.

> The MuJoCo v4 envs are flagged "out of date" by gymnasium; we silence that warning in `helper.silence_env_warnings()`.

---

## File Structure

```
.
├── train.py              # Single-env entry point, argparse
├── train_all.py          # Train pipeline across several envs in sequence
├── train_baselines.py    # Train PPO baselines via RL-Zoo3 for the supported envs
├── evaluate.py           # Pipeline policy vs RL-Zoo3 baseline (true env reward), saves PDF
├── baseline.py           # Load an RL-Zoo3-trained PPO baseline + evaluate on true reward
├── make_plots.py         # CLI: per-env + cross-env PDF plots
├── plots.py              # Plotting functions (read JSON artifacts)
├── paths.py              # Env-first artifact layout helpers
├── trainer.py            # Trainer, TrainerConfig, main training loop, metrics.json
├── ppo_agent.py          # PPOAgent wrapping stable-baselines3 PPO
├── sampler.py            # Trajectory collection, LLM labelling, active learning
├── reward.py             # CompositeReward (alpha * r_fixed + (1-alpha) * R_phi)
├── reward_model.py       # EnsembleRewardModel, PreferenceBuffer, traj_to_segment
├── env_setup.py          # CustomRewardWrapper, eval_with_components, collect_trajectory
├── llm_utils.py          # LLM code generation (reward fn, semantic layer, comparison), caching
├── llm_reflection.py     # ReflectionEngine + reflection_log.json
├── helper.py             # Task loading, env→task map, success criteria, warning suppression
├── prompts/              # reward.md, semantic.md, compare.md, reflection.md
└── tasks/                # pendulum, swimmer, halfcheetah, hopper, walker2d, ant
```

---

## Installation

```bash
conda env create -f environment.yml   # includes rl_zoo3 (baseline training)
conda activate llm-reward
# Torch must be installed manually (see comment in environment.yml), e.g. CPU:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Set your OpenAI key in a `.env` file:

```
OPENAI_API_KEY=sk-...
```

---

## Usage

```bash
# Train one env (task auto-selected from the env via helper.task_for_env)
python train.py --env HalfCheetah-v4 --rounds 9 --queries 10 --ppo-steps 50000 \
    --reward-epochs 50 --lambda-smooth 0.05 --dynamic-batch

# Train the pipeline on all six envs in sequence (each writes to artifacts/<env_id>/...)
python train_all.py --rounds 9 --ppo-steps 50000 --lambda-smooth 0.05 --dynamic-batch

# Train the PPO baselines via RL-Zoo3 (saves to baselines/ppo/<env>_<run>/)
python train_baselines.py                                 # all six envs, tuned budgets
python train_baselines.py --envs HalfCheetah-v4           # a subset

# Evaluate a trained baseline on the true env reward
python baseline.py --env HalfCheetah-v4 --episodes 100

# Evaluate the trained pipeline policy vs the RL-Zoo3 baseline (saves a PDF)
python evaluate.py --env HalfCheetah-v4

# Generate the plot suite (PDFs)
python make_plots.py --env HalfCheetah-v4                 # per-env plots
python make_plots.py --cross-env                          # cross-env bars (all envs)
```

| Argument (`train.py`) | Default | Description |
|---|---|---|
| `--env` | `HalfCheetah-v4` | Environment id |
| `--task` | (auto) | Task file name; defaults to the env's task |
| `--rounds` | 9 | Number of active learning rounds |
| `--queries` | 10 | Preference queries per round |
| `--ppo-steps` | auto | PPO steps per round; default = rl-zoo3 total timesteps for the env ÷ (rounds + 1) |
| `--reward-epochs` | 50 | Reward model training epochs per round |
| `--lambda-smooth` | 1.0 | Temporal smoothness penalty (use `0.05`, see notes) |
| `--dynamic-batch` | off | Scale batch size with buffer size |

---

## Artifact layout (env-first)

Everything an env produces lives under one directory:

```
artifacts/<env_id>/
    policies/policy.zip
    reward_models/{member*.pt, metadata.pt}
    baseline/metrics.json          # RL-Zoo3 baseline returns/lengths (from evaluate.py / baseline.py)
    eval.json                      # evaluate.py's pipeline eval (drives the cross-env plot)
    reflection/{reflection_log.json, reward_round<r>.py, semantic_round<r>.py}
    metrics.json                   # full run config + per-round train/eval data
                                   #   (mean+std+raw per-episode arrays, components,
                                   #    reward-model losses) — everything the plots
                                   #    read, so aesthetics can be re-derived offline
    plots/*.pdf
artifacts/cross_env/*.pdf          # cross-env comparison bars
baselines/ppo/<env_id>_<run>/      # RL-Zoo3 trained baseline (model + VecNormalize)
```

---

## Plots

All plots are saved as **PDF** with clean labels. Per env (`make_plots.py --env <id>`):

- **episode_return_per_round** — eval episodic return (true env reward) vs round (one aggregated point per eval round)
- **reward_model_loss** — CE / smooth / total reward-model loss per round (averaged over that round's epochs)
- **reward_components** — per-component reward sums (mean over eval episodes) across rounds

Cross-env (`make_plots.py --cross-env`): a **single figure** (`artifacts/cross_env/cross_env_comparison.pdf`) with the three metrics as stacked panels sharing the env axis — mean episodic return, mean episode length, and success rate (panel shown only where applicable) — pipeline vs RL-Zoo3 baseline. The numbers come from **`evaluate.py`'s** evaluation: the pipeline bars read each env's `eval.json` and the baseline bars read `baseline/metrics.json`, so **run `python evaluate.py --env <id>` for each env first** (it writes both).

The reflection edit history is **logged** (not plotted): `reflection/reflection_log.json` records, per round, the analysis, reasoning, and whether `r_fixed` / the summarizer were rewritten; each accepted rewrite is also saved as `reward_round<r>.py` / `semantic_round<r>.py`.

---

## Baselines (RL-Zoo3)

We compare against PPO baselines trained with [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo), which ships tuned PPO hyperparameters per env and saves the model **together with its VecNormalize statistics** — so baselines reload faithfully (no warmup hacks).

```bash
python train_baselines.py                       # train all six (tuned budgets)
python train_baselines.py --envs HalfCheetah-v4 --n-timesteps 1000000
python baseline.py --env HalfCheetah-v4          # evaluate on true env reward
```

Trained baselines land in `baselines/ppo/<env_id>_<run>/`. `baseline.py` loads the model + frozen VecNormalize stats and evaluates on the **raw** environment reward (`norm_reward=False`), caching to `artifacts/<env>/baseline/metrics.json` for the plots and `evaluate.py`.

---

## Key Design Decisions

- **LLM as preference labeler** — replaces human annotators; the LLM sees natural-language trajectory descriptions and picks the preferred one
- **Ensemble reward model (×3)** — three independent networks; ensemble variance estimates label uncertainty
- **Active learning** — collect 5× candidate pairs per round, query only the top-`n` by disagreement (cold start uses uniform sampling)
- **Reflection** — after each eval, the LLM may rewrite `r_fixed`/summarizer; accepted reward rewrites trigger a buffer relabel so the BT model stays consistent
- **`lambda_smooth=0.05`** — keep the temporal smoothness penalty small (see notes)
- **Explicit obs layout in task description** — each `tasks/*.txt` annotates observation dimensions (e.g. "obs[8] is forward velocity, not position"); without this the LLM can generate a reward function anti-correlated with the true objective

---

## Notes

**`lambda_smooth`.** The original default `1.0` was too large: the smoothness penalty grew as fast as the CE loss improved, so total loss looked flat (`≈ 3 × ln(2) ≈ 2.08`) while the model was actually learning. With `lambda_smooth=0.05` the CE improvement shows through. `reward_model.train()` returns `(ce_loss, smooth_loss)` separately so both are monitored.

**Obs-layout misreads.** LLMs routinely misinterpret MuJoCo observation vectors (position vs velocity vs angle) because the XML variable names are terse. The annotated obs layout in each task file is a simple, high-leverage fix; alternatively the reflection mechanism can detect and correct a misaligned reward at runtime.

---

## References

- Christiano et al. (2017). [Deep Reinforcement Learning from Human Preferences.](https://arxiv.org/abs/1706.03741) NeurIPS 2017.
- Lee et al. (2021). PEBBLE: Feedback-Efficient Interactive Reinforcement Learning via Relabeling Experience and Unlabeled Data. ICML 2021. *(ensemble reward model design)*
- Ma et al. (2023). EUREKA: Human-Level Reward Design via Coding Large Language Models. *(LLM-as-reward-designer + reflection inspiration)*
- [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) — tuned PPO hyperparameters and training framework used for the baselines.
