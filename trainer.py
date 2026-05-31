from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym

from helper import load_task, section, success_fn_for_env
from llm_utils import cache_key, generate_reward_fn, generate_semantic_fn
from reward_model import RewardModel
from ppo_agent import PPOAgent
from sampler import Sampler
from llm_reflection import ReflectionEngine
from paths import policy_dir, reward_model_dir, reflection_dir, metrics_path

@dataclass
class TrainerConfig:
    env_id: str             = "HalfCheetah-v4"
    task_name: str          = "halfcheetah"
    n_queries: int          = 10        # LLM queries per round
    segment_length: int     = 50        # steps per trajectory segment
    reward_epochs: int      = 50        # reward model training epochs per round
    ppo_steps: int          = 20_000    # PPO timesteps per round
    eval_every: int         = 1         # evaluate every N rounds
    rounds: int             = 9
    artifact_dir: str       = "artifacts"
    progress_bar: bool      = True
    llm_model: str          = "gpt-4o-mini"
    verbose: bool           = True
    lambda_smooth: float    = 1.0         # reward model temporal-smooth penalty
    dynamic_batch: bool     = False       # use max(8,min(32,buffer//2)) vs min(10,buffer)


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
            print(f"Labeller : {cfg.llm_model}")

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
            lambda_smooth = cfg.lambda_smooth,
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
            llm_model      = cfg.llm_model,
            verbose        = cfg.verbose,
            reward_fn      = self.reward_fn,
        )

        # reflection is always on: after each eval the LLM may rewrite r_fixed
        # and the trajectory summarizer based on the round-over-round dynamics.
        composite = self.agent._train_env.envs[0]._reward_fn
        self.reflection = ReflectionEngine(
            task             = self.task,
            reward_code      = reward_code,
            semantic_code    = semantic_code,
            composite_reward = composite,
            sampler          = self.sampler,
            reward_model     = self.reward_model,
            llm_model        = cfg.llm_model,
            verbose          = cfg.verbose,
            output_dir       = reflection_dir(cfg.env_id, cfg.artifact_dir),
        )
        # ── logging ──────────────────────────────────────────
        # eval_rewards / reward_losses / ce_losses / smooth_losses /
        # label_accuracies are kept as flat series for backward compatibility.
        # eval_records / rm_records carry the richer per-round structure dumped
        # to metrics.json for the plotting suite.
        self.eval_rewards:     list[float] = []
        self.reward_losses:    list[float] = []   # total = CE + λ*smooth
        self.ce_losses:        list[float] = []   # CE only
        self.smooth_losses:    list[float] = []   # smooth only (unscaled)
        self.label_accuracies: list[float] = []
        self.eval_records:     list[dict]  = []   # one per eval round
        self.rm_records:       list[dict]  = []   # one per reward-model train step
        self._current_alpha:   float       = 1.0
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
            self._current_alpha = alpha
            self._ppo_step(alpha)
            self._label_step(use_active=True)
            loss = self._reward_model_step(round_label=str(rnd))

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
        self._reward_model_step(round_label="cold")

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

    def _reward_model_step(self, round_label: str = "") -> float:
        n = len(self.reward_model.buffer)
        if self.cfg.dynamic_batch:
            batch_size = max(8, min(32, n // 2)) if n >= 2 else max(1, n)
        else:
            batch_size = min(10, n)
        if self.cfg.verbose:
            print(
                f"  Training reward model | epochs {self.cfg.reward_epochs} "
                f"| batch {batch_size} | labels {n}"
            )
        ce_loss, smooth_loss = self.reward_model.train(
            batch_size=batch_size,
            n_epochs=self.cfg.reward_epochs,
            progress_bar=self.cfg.progress_bar,
        )
        total_loss = ce_loss + self.cfg.lambda_smooth * smooth_loss
        acc = self.reward_model.accuracy()
        self.reward_losses.append(total_loss)
        self.ce_losses.append(ce_loss)
        self.smooth_losses.append(smooth_loss)
        self.label_accuracies.append(acc)
        self.rm_records.append({
            "round":   round_label,
            "ce":      ce_loss,
            "smooth":  smooth_loss,
            "total":   total_loss,
            "acc":     acc,
            "n_labels": n,
        })
        if self.cfg.verbose:
            print(
                f"  Reward model | CE {ce_loss:.4f} | smooth {smooth_loss:.4f}"
                f" | total {total_loss:.4f} | acc {acc:.2%}"
            )
        return total_loss

    def _eval_step(self, rnd: int, loss: float) -> None:
        if self.cfg.verbose:
            print(section(f"Evaluation (round {rnd})"))

        # k-episode evaluation: full per-component breakdown + global metrics
        import numpy as np
        from env_setup import eval_with_components

        n_eval_episodes = 100
        eval_data = eval_with_components(
            self._sample_env,
            self.agent.predict_deterministic,
            self.reward_fn,
            n_episodes=n_eval_episodes,
        )
        ep_rewards = [float(x) for x in eval_data["episode_env_rewards"]]
        ep_lengths = [int(x) for x in eval_data["episode_lengths"]]
        mean_r = float(np.mean(ep_rewards))
        mean_len = float(np.mean(ep_lengths))
        # success rate: prefer env-reported is_success; else a per-env heuristic
        # (e.g. InvertedPendulum survives the full horizon); else N/A.
        sfn = success_fn_for_env(self.cfg.env_id)
        successes = eval_data.get("success")
        if sfn is not None:
            success_rate = float(np.mean([
                1.0 if sfn(l, r) else 0.0
                for l, r in zip(eval_data["episode_lengths"],
                                eval_data["episode_env_rewards"])
            ]))
        elif successes is not None:
            success_rate = float(np.mean([1.0 if s else 0.0 for s in successes]))
        else:
            success_rate = None
        component_sums = {
            k: [float(x) for x in v]
            for k, v in eval_data.get("component_sums", {}).items()
        }
        component_means = {k: float(np.mean(v)) for k, v in component_sums.items()}
        self.eval_rewards.append(mean_r)
        self.eval_records.append({
            "round":              rnd,
            "alpha":              self._current_alpha,
            "n_eval_episodes":    n_eval_episodes,
            # aggregates (used by the default plots)
            "episode_env_reward": mean_r,
            "episode_env_reward_std": float(np.std(ep_rewards)),
            "episode_length":     mean_len,
            "episode_length_std": float(np.std(ep_lengths)),
            "success_rate":       success_rate,
            "component_means":    component_means,
            "loss":               loss,
            "n_labels":           len(self.reward_model.buffer),
            # raw per-episode data (for custom re-plotting: error bars, box plots…)
            "episode_env_rewards": ep_rewards,
            "episode_lengths":     ep_lengths,
            "component_sums":      component_sums,
            "success":             eval_data.get("success"),
        })
        if self.cfg.verbose:
            print(f"  Env reward ({n_eval_episodes} ep mean): {mean_r:.1f} | length: {mean_len:.0f}")

        if self.reflection is not None:
            self.reflection.step(rnd, eval_data, loss, len(self.reward_model.buffer))
            
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
        import json
        from dataclasses import asdict

        reward_dir = reward_model_dir(self.cfg.env_id, self.cfg.artifact_dir)
        pol_dir    = policy_dir(self.cfg.env_id, self.cfg.artifact_dir)
        reward_dir.mkdir(parents=True, exist_ok=True)
        pol_dir.mkdir(parents=True, exist_ok=True)

        self.reward_model.save(str(reward_dir))
        self.agent.save(str(pol_dir / "policy"))

        metrics = {
            "env_id":         self.cfg.env_id,
            "task":           self.cfg.task_name,
            "config":         asdict(self.cfg),   # full run configuration
            "rounds":         self.cfg.rounds,
            "ppo_steps":      self.cfg.ppo_steps,
            "lambda_smooth":  self.cfg.lambda_smooth,
            "eval":           self.eval_records,    # per eval round (+ raw per-episode arrays)
            "reward_model":   self.rm_records,      # per reward-model train step
            "buffer_size":    len(self.reward_model.buffer),
        }
        mpath = metrics_path(self.cfg.env_id, self.cfg.artifact_dir)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(json.dumps(metrics, indent=2))

        if self.cfg.verbose:
            print(section("Artifacts"))
            print(f"  Reward model: {reward_dir}")
            print(f"  Policy      : {pol_dir / 'policy.zip'}")
            print(f"  Metrics     : {mpath}")
