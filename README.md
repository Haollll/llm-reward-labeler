# LLM Reward Labeler v2 — Two-Phase Bradley-Terry Reward Learning

A rebuild of the LLM-reward-labeler idea around an explicit **two-phase**
algorithm. An LLM writes a coded reward function and a trajectory summarizer; the
summarizer feeds an LLM preference labeler that trains a single Bradley-Terry
(BT) reward model over **full trajectories**. PPO is bootstrapped on the coded
reward, then trained on a fixed blend of the coded reward and the learned BT
reward.

This version deliberately **drops** three mechanisms from v1 that complicated
analysis: the ensemble reward model, active/disagreement query selection, and
fixed-length trajectory segments. It also fixes two v1 failure modes: it keeps a
**constant** Phase-II mixing weight (v1 decayed it to ~0 and collapsed onto a
noisy learned reward) and **saves the best-by-eval policy** (v1 saved the last,
which was usually the collapsed one).

**Per-round, on-policy preference data.** Each round rolls out `num_trajs` fresh
trajectories with the *current* policy and compares **all `C(num_trajs, 2)`
pairs**. The BT buffer is **reset every round**, so the reward model is trained
only on this round's on-policy pairs (full-batch); old pairs from a different
policy are discarded.

**PPO hyperparameters come from RL-Zoo3.** `ppo_agent.py` loads each env's tuned
PPO config (learning rate, `n_steps`, `batch_size`, `n_epochs`, `gamma`,
`gae_lambda`, `clip_range`, `ent_coef`, `vf_coef`, `max_grad_norm`,
`policy_kwargs`, `use_sde`), its `n_envs`, and its VecNormalize setting — the
same hyperparameters the baselines were trained with. The coded/BT reward sits
*under* VecNormalize so it always sees raw observations; the policy sees
normalized observations, and eval/collection re-apply the saved obs stats.

## Algorithm

```
Initialize r_fixed, summary_fn (LLM-generated), BT reward model r_model

Phase I  (k1 rounds):           # coded reward + reflection
  PPO(policy, reward=r_fixed, steps=N)
  trajs    = rollout(policy)                      # full episodes
  r_fixed, summary_fn = llm_reflection(summaries(trajs))
  trajs    = relabel_reward_components(trajs, r_fixed)
  labels   = llm_trajectory_comparisons(summaries(trajs, summary_fn))
  BT_train(r_model, labels)                       # cross-entropy over summed rewards

Phase II (k2 rounds):           # mixed reward, no reflection
  reward = alpha * r_fixed + (1 - alpha) * r_model
  PPO(policy, reward=reward, steps=N)
  trajs  = rollout(policy)
  labels = llm_trajectory_comparisons(summaries(trajs, summary_fn))
  BT_train(r_model, labels)
```

## Files

| File | Role |
|------|------|
| `llm.py` | env description, LLM code-gen (reward + summarizer), preference comparison, Phase-I reflection |
| `reward_model.py` | single-network BT model over full (padded+masked) trajectories; per-epoch loss; normalized predict |
| `reward_fn.py` | `CompositeReward` = `alpha*r_fixed + (1-alpha)*R_phi` (constant alpha) |
| `env_utils.py` | reward wrapper, full-episode collection, component relabel, 100-ep evaluation |
| `zoo_hyperparams.py` | load + translate RL-Zoo3's tuned PPO hyperparameters for an env |
| `ppo_agent.py` | stable-baselines3 PPO wrapper (uses zoo hyperparameters + VecNormalize) |
| `trainer.py` | the two-phase loop + best-policy saving + metrics dump |
| `train.py` / `train_all.py` | single-env / multi-env entry points |
| `baseline.py` | load + evaluate the reused RL-Zoo3 PPO baselines |
| `evaluate.py` | 100-ep pipeline eval, baseline eval, single-episode BT-vs-env series |
| `plots.py` / `make_plots.py` | the four result plots |
| `prompts/` | `reward.md`, `semantic.md`, `compare.md`, `reflection.md` |
| `tasks/` | per-env observation-layout descriptions (shared with v1) |
| `baselines/` | symlink to v1's RL-Zoo3 baselines (no retraining) |

## Usage

```bash
# one env
python train.py --env HalfCheetah-v4 --k1 5 --k2 4 --ppo-steps 100000
python evaluate.py --env HalfCheetah-v4
python make_plots.py --env HalfCheetah-v4

# everything (6 envs, train → eval → plot)
./run_experiment.sh
```

Requires `OPENAI_API_KEY` in `.env`. Install deps from `environment.yml` (then
install torch separately, see the file's footer).

## Plots (per env, under `artifacts/<env>/plots/`)

1. **`training_phases.pdf`** — mean 100-episode return per round, Phase I and II
   on one axis with a dashed vertical line at the phase boundary.
2. **`bt_loss_per_epoch.pdf`** — BT cross-entropy loss at every epoch of every
   round, concatenated across both phases (≈ `epochs × rounds` points), with
   round guides and the phase boundary marked.
3. **`bt_vs_env.pdf`** — per-timestep BT model reward vs true env reward for a
   single baseline-policy episode (twin y-axes; Pearson r in the title).
4. **`eval_metrics_bars.pdf`** — 100-episode mean return, mean episode length,
   and success rate (when applicable), pipeline vs baseline.

## Data artifacts (per env)

* `metrics.json` — per-round records: phase, alpha, mean/std return, episode
  length, component means, **per-epoch BT loss list**, BT accuracy, buffer size.
* `eval.json` — 100-episode pipeline evaluation (return/length/success arrays).
* `baseline/metrics.json` — RL-Zoo3 baseline evaluation.
* `bt_vs_env.json` — per-step env reward + BT reward for one episode.
* `reflection/` — per-round reflection log + versioned `reward_round*.py` /
  `semantic_round*.py` rewrites.
