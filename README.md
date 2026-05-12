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
    └── halfcheetah.txt   # Task description for HalfCheetah-v5
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

# Full run with LLM as preference labeler
python train.py --rounds 9 --queries 10 --ppo-steps 50000

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
| `--oracle` | off | Use ground-truth reward sums instead of LLM labels |

---

## Key Design Decisions

- **LLM as preference labeler** — replaces human annotators; the LLM sees natural-language trajectory descriptions and picks the preferred one
- **Ensemble reward model (×3)** — three independent networks trained on the same preference buffer; ensemble variance estimates label uncertainty
- **Active learning** — collect 5× candidate trajectory pairs per round, score each by ensemble disagreement, query only the top-`n` by disagreement (cold start round uses uniform sampling)
- **`CustomRewardWrapper`** — backward-compatible; `reward_model=None` uses only `r_fixed`, making it easy to ablate the learned component
- **LLM output caching** — generated reward and semantic functions are cached to disk, avoiding redundant API calls across runs

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

## References

- Christiano et al. (2017). [Deep Reinforcement Learning from Human Preferences.](https://arxiv.org/abs/1706.03741) NeurIPS 2017.
- Lee et al. (2021). PEBBLE: Feedback-Efficient Interactive Reinforcement Learning via Relabeling Experience and Unlabeled Data. ICML 2021. *(ensemble reward model design)*
- Ma et al. (2023). EUREKA: Human-Level Reward Design via Coding Large Language Models. *(LLM-as-reward-designer inspiration)*
