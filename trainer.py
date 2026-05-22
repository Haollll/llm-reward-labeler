from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import gymnasium as gym

from helper import load_task, section
from llm_utils import cache_key, generate_reward_fn, generate_semantic_fn
from reward_model import RewardModel
from ppo_agent import PPOAgent
from sampler import Sampler
from llm_reflection import ReflectionEngine

@dataclass
class TrainerConfig:
    env_id: str             = "HalfCheetah-v5"
    task_name: str          = "halfcheetah"
    use_oracle: bool        = False
    n_queries: int          = 10        # LLM queries per round
    segment_length: int     = 50        # steps per trajectory segment
    reward_epochs: int      = 50        # reward model training epochs per round
    ppo_steps: int          = 20_000    # PPO timesteps per round
    eval_every: int         = 2         # evaluate every N rounds
    rounds: int             = 9
    artifact_dir: str       = "artifacts"
    progress_bar: bool      = True
    llm_model: str          = "gpt-4o-mini"
    verbose: bool           = True


# ─────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────

class Trainer:

    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg

        # ── task ────────────────────────────────────────────
        self.task = load_task(cfg.task_name)

        if cfg.verbose:
            print(section("Setup"))
            print(f"Env      : {cfg.env_id}")
            print(f"Task     : {cfg.task_name}")
            print(f"Labeller : {'oracle' if cfg.use_oracle else cfg.llm_model}")

        # ── LLM-generated components ────────────────────────
        _probe_env = gym.make(cfg.env_id)
        if cfg.verbose:
            print(section("Generating reward fn + semantic layer"))

        reward_code, self.reward_fn   = generate_reward_fn(_probe_env, self.task, model=cfg.llm_model)
        semantic_code, self.semantic_fn = generate_semantic_fn(_probe_env, self.task, model=cfg.llm_model)
        self.cache_id = cache_key(_probe_env, self.task)
        _probe_env.close()

        if cfg.verbose:
            print("  ✓ cached for future runs")

        # ── reward model ────────────────────────────────────
        _env = gym.make(cfg.env_id)
        self.reward_model = RewardModel(
            env           = _env,
            size_segment  = cfg.segment_length,
        )
        _env.close()

        # ── PPO agent ────────────────────────────────────────
        self.agent = PPOAgent(
            env_id        = cfg.env_id,
            reward_fn     = self.reward_fn,
            reward_model  = self.reward_model,
            cache_key     = self.cache_id,
            progress_bar  = cfg.progress_bar,
            verbose       = 0,
        )

        # ── sampler ──────────────────────────────────────────
        # sampler uses a separate env instance for trajectory collection
        self._sample_env = gym.make(cfg.env_id)
        self.sampler = Sampler(
            env            = self._sample_env,
            policy_fn      = self.agent.predict,
            reward_model   = self.reward_model,
            semantic_fn    = self.semantic_fn,
            task           = self.task,
            segment_length = cfg.segment_length,
            use_oracle     = cfg.use_oracle,
            llm_model      = cfg.llm_model,
            verbose        = cfg.verbose,
        )
        self.reflection: Optional[ReflectionEngine] = None
        
        if not cfg.use_oracle:
            composite = self.agent._train_env.envs[0]._reward_fn
            self.reflection = ReflectionEngine(
                task             = self.task,
                reward_code      = reward_code,
                semantic_code    = semantic_code,
                composite_reward = composite,
                sampler          = self.sampler,
                llm_model        = cfg.llm_model,
                verbose          = cfg.verbose,
            )
        # ── logging ──────────────────────────────────────────
        self.eval_rewards:     list[float] = []
        self.reward_losses:    list[float] = []
        self.label_accuracies: list[float] = []
        self.artifact_root = Path(cfg.artifact_dir)

    # ── public ───────────────────────────────────────────────

    def run(self, n_rounds: int) -> None:
        if self.cfg.verbose:
            print(section("Training"))
            print(f"Rounds {n_rounds} | queries/round {self.cfg.n_queries} "
                  f"| PPO steps/round {self.cfg.ppo_steps}")

        self._cold_start()

        for rnd in range(1, n_rounds + 1):
            if self.cfg.verbose:
                print(section(f"Round {rnd}/{n_rounds}"))

            alpha = self._alpha_for_round(rnd, n_rounds)
            self._ppo_step(alpha)
            self._label_step(use_active=True)
            loss = self._reward_model_step()

            if rnd % self.cfg.eval_every == 0:
                self._eval_step(rnd, loss)

        self._print_summary()
        self._save_artifacts()

    # ── private: training steps ──────────────────────────────

    def _cold_start(self) -> None:
        """Uniform sampling before reward model is trained."""
        if self.cfg.verbose:
            print(section("Cold start"))
        self._label_step(use_active=False)
        self._reward_model_step()

    def _alpha_for_round(self, rnd: int, n_rounds: int) -> float:
        if n_rounds <= 1:
            return 1.0
        return 1.0 - ((rnd - 1) / (n_rounds - 1))

    def _ppo_step(self, alpha: float) -> None:
        if self.cfg.verbose:
            print(f"  Training PPO policy | steps {self.cfg.ppo_steps} | alpha {alpha:.3f}")
        self.agent.set_alpha(alpha)
        self.agent.train(total_timesteps=self.cfg.ppo_steps)
        if self.cfg.verbose:
            print("  PPO policy training complete")

    def _label_step(self, use_active: bool) -> None:
        added = self.sampler.collect_and_label(
            n_queries=self.cfg.n_queries,
            use_active=use_active,
        )
        if self.cfg.verbose:
            print(f"  Labels added: {added} | buffer: {len(self.reward_model.buffer)}")

    def _reward_model_step(self) -> float:
        batch_size = min(64, len(self.reward_model.buffer))
        if self.cfg.verbose:
            print(
                f"  Training reward model | epochs {self.cfg.reward_epochs} "
                f"| batch {batch_size} | labels {len(self.reward_model.buffer)}"
            )
        loss = self.reward_model.train(
            batch_size=batch_size,
            n_epochs=self.cfg.reward_epochs,
            progress_bar=self.cfg.progress_bar,
        )
        acc = self.reward_model.accuracy()
        self.reward_losses.append(loss)
        if self.cfg.verbose:
            print(f"  Reward model | loss {loss:.4f} | acc {acc:.2%}")
        return loss

    def _eval_step(self, rnd: int, loss: float) -> None:
        if self.cfg.verbose:
            print(section(f"Evaluation (round {rnd})"))

        mean_r = self.agent.evaluate(n_episodes=5)
        self.eval_rewards.append(mean_r)
        if self.cfg.verbose:
            print(f"  True reward (5 ep): {mean_r:.1f}")

        acc = self.sampler.measure_label_accuracy(n_pairs=15)
        self.label_accuracies.append(acc)
        if self.cfg.verbose:
            print(f"  LLM label accuracy: {acc:.1%}")

        if self.reflection is not None:
            self.reflection.step(rnd, mean_r, loss, acc, len(self.reward_model.buffer))
            
    # ── private: summary ─────────────────────────────────────

    def _print_summary(self) -> None:
        if self.cfg.verbose:
            print(section("Summary"))
        if self.eval_rewards:
            print(f"  Final reward  : {self.eval_rewards[-1]:.1f}")
            print(f"  Best reward   : {max(self.eval_rewards):.1f}")
        if self.label_accuracies:
            print(f"  LLM accuracy  : {self.label_accuracies[-1]:.1%}")
        print(f"  Total labels  : {len(self.reward_model.buffer)}")

    def _save_artifacts(self) -> None:
        reward_dir = self.artifact_root / "reward_models" / self.cache_id
        policy_dir = self.artifact_root / "policies" / self.cache_id
        reward_dir.mkdir(parents=True, exist_ok=True)
        policy_dir.mkdir(parents=True, exist_ok=True)

        self.reward_model.save(str(reward_dir))
        self.agent.save(str(policy_dir / "policy"))

        if self.cfg.verbose:
            print(section("Artifacts"))
            print(f"  Reward model: {reward_dir}")
            print(f"  Policy      : {policy_dir / 'policy.zip'}")
