# CSE 579 Project: LLM-Labeled RLHF on HalfCheetah-v5

Use GPT-4o-mini as a drop-in replacement for human annotators in Reinforcement Learning from Human Feedback (RLHF), training a reward model on LLM-generated preference labels.

Based on: Christiano et al., *Deep Reinforcement Learning from Human Preferences*, NeurIPS 2017.

---

## Project Status

| File | Status | Description |
|------|--------|-------------|
| `env_setup.py` | **DONE** | HalfCheetah env, feature extraction, trajectory-to-text |
| `llm_labeler.py` | PLANNED | GPT-4o-mini labels trajectory pairs via Bradley-Terry pairwise comparison |
| `reward_model.py` | PLANNED | Ensemble reward model trained on LLM preference labels |
| `train.py` | PLANNED | Full training loop with online query collection |

---

## Architecture

```
HalfCheetah-v5 (MuJoCo)
        |
        v
  collect_trajectory()          env_setup.py
        |
        v
  extract_features()            env_setup.py
  features_to_text()
        |
        v
  GPT-4o-mini pairwise label    llm_labeler.py  (PLANNED)
  (Bradley-Terry comparison)
        |
        v
  Ensemble Reward Model R_phi   reward_model.py (PLANNED)
        |
        v
  Full Reward:                  train.py        (PLANNED)
  r_t = r_fixed(t) + g(s,a,s') * R_phi(s,a,s')
        |
        v
  Policy Optimization (PPO/SAC)
```

---

## Reward Formulation

```
r_t = r_fixed(t) + g(s_t, a_t, s_{t+1}) * R_phi(s_t, a_t, s_{t+1})
```

- `r_fixed`: environment reward signal
- `g(·)`: gating function
- `R_phi`: learned reward model from LLM preference labels

---

## Setup

```bash
conda env create -f environment.yml
conda activate cse579
```

**Device:** auto-detected at runtime — CUDA > MPS > CPU.

---

## Reference

Christiano, P., Leike, J., Brown, T. B., Martic, M., Legg, S., & Amodei, D. (2017).
*Deep Reinforcement Learning from Human Preferences.* NeurIPS 2017.
