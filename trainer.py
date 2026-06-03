"""Two-phase trainer for v2.

Phase I  (k1 rounds): PPO optimizes the coded reward r_fixed only. After each
round the LLM reflects on the training feedback and may rewrite r_fixed and the
trajectory summarizer; the collected trajectories are relabelled under the new
r_fixed, summarized, compared by the LLM, and used to train the Bradley-Terry
reward model.

Phase II (k2 rounds): PPO optimizes a fixed blend alpha*r_fixed + (1-alpha)*R_phi
(no more reflection). Trajectories are still collected, compared, and used to
keep training the BT model.

Differences from v1 (deliberate): single BT network, full trajectories (no
segments), uniform random pairs (no active learning), constant Phase-II alpha
(no decay to 0), and the BEST-by-eval policy is saved (not the last).
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import gymnasium as gym

from helper import load_task, section, success_fn_for_env
from llm import (
    cache_key, generate_reward_fn, generate_semantic_fn,
    compare_trajectories, reflect, exec_fn, strip_fences,
)
from reward_model import BTRewardModel, traj_to_sa
from ppo_agent import PPOAgent
from env_utils import (
    collect_trajectory, relabel_reward_components, eval_with_components,
)
from paths import (
    policy_dir, reward_model_dir, reflection_dir, metrics_path,
)


@dataclass
class TrainerConfig:
    env_id: str        = "HalfCheetah-v4"
    task_name: str     = "halfcheetah"
    k1: int            = 5          # Phase I rounds
    k2: int            = 5          # Phase II rounds
    ppo_steps: int     = 100_000    # PPO timesteps per round (N)
    num_trajs: int     = 10         # trajectories collected per round; all
                                    # C(num_trajs, 2)=45 pairs are compared+trained
                                    # on this round only (buffer reset each round)
    reward_epochs: int = 50         # BT training epochs per round
    alpha: float       = 0.5        # constant Phase-II mixing weight
    eval_episodes: int = 100
    reflect: bool      = True       # Phase-I reflection on/off
    llm_model: str     = "gpt-4o-mini"
    reflect_model: str = "gpt-4o"
    artifact_dir: str  = "artifacts"
    progress_bar: bool = True
    verbose: bool      = True


class Trainer:
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        self.task = load_task(cfg.task_name)

        if cfg.verbose:
            print(section("Setup"))
            print(f"Env      : {cfg.env_id}")
            print(f"Task     : {cfg.task_name}")
            print(f"Phases   : I={cfg.k1} rounds (coded reward), "
                  f"II={cfg.k2} rounds (alpha={cfg.alpha} blend)")
            print(f"Labeller : {cfg.llm_model} | Reflector: {cfg.reflect_model}")

        # ── LLM-generated reward + summarizer ────────────────
        probe = gym.make(cfg.env_id)
        if cfg.verbose:
            print(section("Generating coded reward + semantic summarizer"))
        self.reward_code, self.reward_fn = generate_reward_fn(probe, self.task, cfg.llm_model)
        self.semantic_code, self.semantic_fn = generate_semantic_fn(probe, self.task, cfg.llm_model)
        self.cache_id = cache_key(probe, self.task)
        probe.close()

        # ── BT reward model ──────────────────────────────────
        rm_env = gym.make(cfg.env_id)
        self.reward_model = BTRewardModel(rm_env)
        rm_env.close()

        # ── PPO agent (composite reward; reward_model attached) ──
        self.agent = PPOAgent(
            env_id=cfg.env_id,
            reward_fn=self.reward_fn,
            reward_model=self.reward_model,
            progress_bar=cfg.progress_bar,
            verbose=0,
        )
        # keep the live composite in sync when r_fixed is hot-swapped
        self._composite = self.agent._composite

        # ── env for collection / evaluation ──────────────────
        self._sample_env = gym.make(cfg.env_id)

        # ── logging ──────────────────────────────────────────
        self.round_records: List[dict] = []
        self.snapshots: List[dict] = []        # for Phase-I reflection
        self.reflection_log: List[dict] = []
        self.best_return = -np.inf
        self._global_round = 0

    # ─────────────────────────────────────────────────────────
    # Public
    # ─────────────────────────────────────────────────────────
    def run(self) -> None:
        if self.cfg.verbose:
            print(section("Phase I — coded reward + reflection"))
        for r in range(1, self.cfg.k1 + 1):
            self._phase_round(phase="I", phase_round=r)

        # entering Phase II: BT model now drives part of the reward
        self.agent.attach_reward_model(self.reward_model)
        self.agent.set_alpha(self.cfg.alpha)
        if self.cfg.verbose:
            print(section("Phase II — mixed reward (no reflection)"))
        for r in range(1, self.cfg.k2 + 1):
            self._phase_round(phase="II", phase_round=r)

        self._print_summary()
        self._save_artifacts()

    # ─────────────────────────────────────────────────────────
    # One round (shared structure for both phases)
    # ─────────────────────────────────────────────────────────
    def _phase_round(self, phase: str, phase_round: int) -> None:
        self._global_round += 1
        alpha = 1.0 if phase == "I" else self.cfg.alpha
        if self.cfg.verbose:
            print(section(f"Phase {phase} · round {phase_round} "
                          f"(global {self._global_round}) · alpha {alpha:.2f}"))

        # 1. PPO with the current reward
        if self.cfg.verbose:
            print(f"  PPO | {self.cfg.ppo_steps} steps | alpha {alpha:.2f}")
        self.agent.train(self.cfg.ppo_steps)

        # 2. Evaluate on TRUE env reward (drives the per-round plot + reflection)
        eval_data = self._evaluate()
        self._record_eval(phase, alpha, eval_data)

        # 3. Collect trajectories with the current policy
        trajs = self._collect(self.cfg.num_trajs)

        # 4. Phase-I only: reflect → maybe rewrite r_fixed & summarizer
        if phase == "I" and self.cfg.reflect:
            self._reflect_step()
            # 5. relabel reward components under the (possibly) new r_fixed
            trajs = relabel_reward_components(trajs, self.reward_fn)

        # 6. Semantic summaries + LLM comparisons → preference labels
        added = self._label_and_store(trajs)

        # 7. Bradley-Terry training
        losses = self.reward_model.train(n_epochs=self.cfg.reward_epochs)
        acc = self.reward_model.accuracy()
        self.round_records[-1].update({
            "bt_loss_epochs": [float(x) for x in losses],
            "bt_loss_mean": float(np.mean(losses)) if losses else 0.0,
            "bt_acc": acc,
            "n_labels": len(self.reward_model.buffer),
            "labels_added": added,
        })
        if self.cfg.verbose:
            print(f"  BT model | epochs {self.cfg.reward_epochs} | "
                  f"loss {np.mean(losses):.4f} | acc {acc:.2%} | "
                  f"buffer {len(self.reward_model.buffer)}")

    # ─────────────────────────────────────────────────────────
    # Steps
    # ─────────────────────────────────────────────────────────
    def _collect(self, n: int) -> List[List[Tuple]]:
        return [
            collect_trajectory(self._sample_env, self.agent.predict, reward_fn=self.reward_fn)
            for _ in range(n)
        ]

    def _make_pairs(self, n_trajs: int) -> List[Tuple[int, int]]:
        """All unique unordered pairs — C(n_trajs, 2) comparisons."""
        from itertools import combinations
        return [(a, b) for a, b in combinations(range(n_trajs), 2)]

    def _label_and_store(self, trajs: List[List[Tuple]]) -> int:
        if len(trajs) < 2:
            return 0
        # Train the BT model on THIS round's trajectories only: the policy that
        # rolled them out is different every round, so old pairs are off-policy.
        self.reward_model.buffer.clear()
        added = 0
        for a, b in self._make_pairs(len(trajs)):
            try:
                label, expl = compare_trajectories(
                    trajs[a], trajs[b], self.semantic_fn, self.task, self.cfg.llm_model)
            except Exception as e:
                if self.cfg.verbose:
                    print(f"    [compare skipped] {type(e).__name__}: {e}")
                continue
            self.reward_model.add_pair(traj_to_sa(trajs[a]), traj_to_sa(trajs[b]), float(label))
            added += 1
            if self.cfg.verbose:
                print(f"    LLM → {'A' if label == 1 else 'B'} | {expl[:70]}")
        return added

    def _evaluate(self) -> dict:
        return eval_with_components(
            self._sample_env, self.agent.predict_deterministic,
            self.reward_fn, n_episodes=self.cfg.eval_episodes)

    def _record_eval(self, phase: str, alpha: float, eval_data: dict) -> None:
        ep_rewards = [float(x) for x in eval_data["episode_env_rewards"]]
        ep_lengths = [int(x) for x in eval_data["episode_lengths"]]
        mean_r = float(np.mean(ep_rewards))
        mean_len = float(np.mean(ep_lengths))
        component_sums = {k: [float(x) for x in v]
                          for k, v in eval_data.get("component_sums", {}).items()}
        component_means = {k: float(np.mean(v)) for k, v in component_sums.items()}

        self.round_records.append({
            "phase": phase,
            "global_round": self._global_round,
            "alpha": alpha,
            "episode_env_reward": mean_r,
            "episode_env_reward_std": float(np.std(ep_rewards)),
            "episode_length": mean_len,
            "component_means": component_means,
            "episode_env_rewards": ep_rewards,
        })
        # snapshot for reflection (Phase I)
        self.snapshots.append({
            "component_means": component_means,
            "episode_length": mean_len,
            "episode_env_reward": mean_r,
        })
        if self.cfg.verbose:
            print(f"  Eval | {self.cfg.eval_episodes} ep | return {mean_r:.1f} | "
                  f"length {mean_len:.0f}")

        # save the BEST-by-eval policy (v1 saved the last, which collapsed)
        if mean_r > self.best_return:
            self.best_return = mean_r
            pol = policy_dir(self.cfg.env_id, self.cfg.artifact_dir)
            pol.mkdir(parents=True, exist_ok=True)
            self.agent.save(str(pol / "policy"))
            if self.cfg.verbose:
                print(f"  ✓ new best policy ({mean_r:.1f}) saved")

    # ── Phase-I reflection ───────────────────────────────────
    def _reflect_step(self) -> None:
        if self.cfg.verbose:
            print("  [Reflection] calling LLM...")
        try:
            result = reflect(self.task, self.reward_code, self.semantic_code,
                             self.snapshots, model=self.cfg.reflect_model)
        except Exception as e:
            print(f"  [reflection skipped] {type(e).__name__}: {e}")
            return

        reward_swapped = self._maybe_swap(result.get("reward_code"), "reward")
        semantic_swapped = self._maybe_swap(result.get("semantic_code"), "summarize")
        self.reflection_log.append({
            "global_round": self._global_round,
            "analysis": result.get("analysis"),
            "reasoning": result.get("reasoning"),
            "reward_swapped": reward_swapped,
            "semantic_swapped": semantic_swapped,
        })
        rdir = reflection_dir(self.cfg.env_id, self.cfg.artifact_dir)
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "reflection_log.json").write_text(json.dumps(self.reflection_log, indent=2))
        if self.cfg.verbose:
            print(f"  → reward_swapped={reward_swapped} semantic_swapped={semantic_swapped}")

    def _maybe_swap(self, code: Optional[str], fn_name: str) -> bool:
        if not code:
            return False
        code = strip_fences(code)
        try:
            fn = exec_fn(code, fn_name)
            if not self._smoke_test(fn, fn_name):
                return False
        except Exception as e:
            print(f"  Warning: {fn_name} compile/smoke failed ({e}); keeping current")
            return False

        rdir = reflection_dir(self.cfg.env_id, self.cfg.artifact_dir)
        rdir.mkdir(parents=True, exist_ok=True)
        if fn_name == "reward":
            self.reward_fn = fn
            self.reward_code = code
            self._composite.r_fixed = fn        # live PPO env picks it up
            (rdir / f"reward_round{self._global_round}.py").write_text(code)
        else:
            self.semantic_fn = fn
            self.semantic_code = code
            (rdir / f"semantic_round{self._global_round}.py").write_text(code)
        return True

    def _smoke_test(self, fn, fn_name: str) -> bool:
        env = self._sample_env
        obs = env.observation_space.sample()
        action = env.action_space.sample()
        try:
            if fn_name == "reward":
                out = fn(obs, action, obs)
                vals = list(out.values()) if isinstance(out, dict) else [out]
                return all(np.isfinite(float(v)) for v in vals)
            else:
                step = (obs, action, obs, {"total": 0.0, "dummy": 0.0}, False)
                return isinstance(fn([step, step]), str)
        except Exception as e:
            print(f"  Warning: {fn_name} smoke test raised {type(e).__name__}: {e}")
            return False

    # ── summary + save ───────────────────────────────────────
    def _print_summary(self) -> None:
        if self.cfg.verbose:
            print(section("Summary"))
        rets = [r["episode_env_reward"] for r in self.round_records]
        if rets:
            print(f"  Final round return : {rets[-1]:.1f}")
            print(f"  Best  round return : {max(rets):.1f} (saved policy)")
        print(f"  Buffer size        : {len(self.reward_model.buffer)}")

    def _save_artifacts(self) -> None:
        rmdir = reward_model_dir(self.cfg.env_id, self.cfg.artifact_dir)
        self.reward_model.save(str(rmdir))
        # policy.zip already saved (best-by-eval) during training

        metrics = {
            "env_id": self.cfg.env_id,
            "task": self.cfg.task_name,
            "config": asdict(self.cfg),
            "k1": self.cfg.k1,
            "k2": self.cfg.k2,
            "alpha": self.cfg.alpha,
            "reward_epochs": self.cfg.reward_epochs,
            "rounds": self.round_records,
            "buffer_size": len(self.reward_model.buffer),
            "best_return": float(self.best_return),
        }
        mpath = metrics_path(self.cfg.env_id, self.cfg.artifact_dir)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(json.dumps(metrics, indent=2))
        if self.cfg.verbose:
            print(section("Artifacts"))
            print(f"  Reward model : {rmdir}")
            print(f"  Policy       : {policy_dir(self.cfg.env_id, self.cfg.artifact_dir) / 'policy.zip'}")
            print(f"  Metrics      : {mpath}")
