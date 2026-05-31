from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI

from llm_utils import _exec_fn, _load_prompt, CACHE_DIR

client = OpenAI()


# ── data ─────────────────────────────────────────────────────

@dataclass
class RoundSnapshot:
    """One per round, post-PPO. Records what the policy looks like *now*."""
    rnd:                int
    component_means:    Dict[str, float]      # mean over eval episodes of per-trajectory component sums
    episode_length:     float                 # mean episode length over eval episodes
    episode_env_reward: float                 # mean per-episode env reward over eval episodes
    success_rate:       Optional[float]       # mean over eval episodes, None if env never reports is_success
    loss:               float
    n_labels:           int


# ── status formatting ─────────────────────────────────────────

def _format_series(name: str, vals: List[float], fmt: str) -> str:
    quoted = ", ".join(f"'{v:{fmt}}'" for v in vals)
    mx = max(vals)
    mn = min(vals)
    mu = sum(vals) / len(vals)
    return f"{name}: [{quoted}], Max: {mx:{fmt}}, Mean: {mu:{fmt}}, Min: {mn:{fmt}}"


def _format_training_history(snapshots: List[RoundSnapshot]) -> str:
    if not snapshots:
        return "No training data yet."

    lines: List[str] = []

    # per-component series — sorted name for stable ordering
    all_comps: set = set()
    for s in snapshots:
        all_comps.update(s.component_means.keys())
    for comp in sorted(all_comps):
        vals = [s.component_means.get(comp, 0.0) for s in snapshots]
        lines.append(_format_series(comp, vals, ".3f"))

    # success rate — only if at least one snapshot reported a non-None value
    success_vals = [s.success_rate for s in snapshots if s.success_rate is not None]
    if success_vals:
        # backfill missing entries with 0.0 to keep alignment with the round index
        aligned = [s.success_rate if s.success_rate is not None else 0.0 for s in snapshots]
        lines.append(_format_series("success_rate", aligned, ".3f"))

    # episode length and episode env reward — always reported
    lines.append(_format_series(
        "episode_length", [s.episode_length for s in snapshots], ".1f",
    ))
    lines.append(_format_series(
        "episode_env_reward", [s.episode_env_reward for s in snapshots], ".2f",
    ))

    return "\n".join(lines)


# ── prompt assembly ───────────────────────────────────────────

def _build_prompt(task: str, reward_code: str, semantic_code: str, training_summary: str) -> str:
    return _load_prompt("reflection").format(
        task             = task,
        reward_code      = reward_code,
        semantic_code    = semantic_code,
        training_summary = training_summary,
    )


# ── engine ───────────────────────────────────────────────────

class ReflectionEngine:
    """
    EUREKA-style reflection: after each round, summarise the training dynamics
    (per-component values + global metrics across rounds) and let the LLM
    rewrite both `r_fixed` and `summarize` based on that feedback.
    """

    def __init__(
        self,
        task: str,
        reward_code: str,
        semantic_code: str,
        composite_reward,
        sampler,
        reward_model = None,
        reflect_every: int = 1,
        llm_model:    str  = "gpt-4o",
        verbose:      bool = True,
        output_dir = None,
    ):
        self.task           = task
        self.reward_code    = reward_code
        self.semantic_code  = semantic_code
        self.composite      = composite_reward
        self.sampler        = sampler
        self.reward_model   = reward_model
        self.reflect_every  = reflect_every
        self.llm_model      = llm_model
        self.verbose        = verbose
        # where versioned reward/semantic snapshots and the reflection log are
        # written; falls back to the global LLM cache dir if unset.
        self.output_dir     = Path(output_dir) if output_dir is not None else CACHE_DIR

        self.snapshots: List[RoundSnapshot] = []
        self.log:       List[dict]          = []
        self._n_evals:  int                 = 0

    # ── public ───────────────────────────────────────────────

    def step(
        self,
        rnd: int,
        eval_data: Dict[str, Any],
        loss: float,
        n_labels: int,
    ) -> Optional[dict]:
        """Record a snapshot from this round's eval and reflect if due."""
        snap = self._build_snapshot(rnd, eval_data, loss, n_labels)
        self.snapshots.append(snap)
        self._n_evals += 1

        if self.verbose:
            sr = f"{snap.success_rate:.2%}" if snap.success_rate is not None else "n/a"
            print(
                f"  Snapshot | env_reward {snap.episode_env_reward:+.1f} | "
                f"length {snap.episode_length:.0f} | success {sr}"
            )

        if self._n_evals % self.reflect_every == 0:
            return self._reflect()
        return None

    # ── private: snapshot ────────────────────────────────────

    def _build_snapshot(
        self,
        rnd: int,
        eval_data: Dict[str, Any],
        loss: float,
        n_labels: int,
    ) -> RoundSnapshot:
        component_means = {
            k: float(np.mean(v)) for k, v in eval_data.get("component_sums", {}).items()
        }
        successes = eval_data.get("success")
        success_rate = (
            float(np.mean([1.0 if s else 0.0 for s in successes]))
            if successes is not None
            else None
        )
        return RoundSnapshot(
            rnd                = rnd,
            component_means    = component_means,
            episode_length     = float(np.mean(eval_data["episode_lengths"])),
            episode_env_reward = float(np.mean(eval_data["episode_env_rewards"])),
            success_rate       = success_rate,
            loss               = loss,
            n_labels           = n_labels,
        )

    # ── private: LLM call ────────────────────────────────────

    def _reflect(self) -> dict:
        if self.verbose:
            print("\n  [Reflection] Calling LLM...")

        training_summary = _format_training_history(self.snapshots)
        prompt = _build_prompt(
            task             = self.task,
            reward_code      = self.reward_code,
            semantic_code    = self.semantic_code,
            training_summary = training_summary,
        )

        resp = client.chat.completions.create(
            model           = self.llm_model,
            messages        = [{"role": "user", "content": prompt}],
            response_format = {"type": "json_object"},
            temperature     = 0.3,
        )
        result = json.loads(resp.choices[0].message.content)
        rnd = self.snapshots[-1].rnd
        result["round"] = rnd

        if self.verbose:
            analysis = (result.get("analysis") or "")[:240]
            print(f"  → {analysis}")

        reward_swapped, semantic_swapped = self._apply(result, rnd)

        # structured log entry: what the LLM proposed and what actually took.
        self.log.append({
            "round":            rnd,
            "analysis":         result.get("analysis"),
            "reasoning":        result.get("reasoning"),
            "reward_proposed":  bool(result.get("reward_code")),
            "reward_swapped":   reward_swapped,
            "semantic_proposed": bool(result.get("semantic_code")),
            "semantic_swapped": semantic_swapped,
            "reward_code":      self.reward_code if reward_swapped else None,
            "semantic_code":    self.semantic_code if semantic_swapped else None,
        })
        self._write_log()
        return result

    # ── private: apply ───────────────────────────────────────

    def _apply(self, result: dict, rnd: int) -> tuple[bool, bool]:
        reward_code   = result.get("reward_code")
        semantic_code = result.get("semantic_code")

        reward_swapped = semantic_swapped = False
        if reward_code:
            reward_swapped = self._hot_swap(reward_code, "reward", rnd)
        if semantic_code:
            semantic_swapped = self._hot_swap(semantic_code, "summarize", rnd)

        # r_fixed changed → buffer's labels were assigned under the previous
        # r_fixed and are now stale. Relabel with the new function so the BT
        # model is trained against consistent taste.
        if reward_swapped and self.reward_model is not None:
            n = self.reward_model.relabel_buffer(self.composite.r_fixed)
            if self.verbose:
                print(f"  ✓ buffer relabelled under new r_fixed ({n} pairs)")
        return reward_swapped, semantic_swapped

    def _write_log(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "reflection_log.json").write_text(
            json.dumps(self.log, indent=2)
        )

    def _smoke_test(self, fn, fn_name: str) -> bool:
        """Verify a freshly compiled fn runs cleanly on a sample input and
        does not return NaN/Inf. Returns True if safe to install."""
        env = self.sampler.env
        try:
            obs    = env.observation_space.sample()
            action = env.action_space.sample()

            if fn_name == "reward":
                out = fn(obs, action, obs)
                if isinstance(out, dict):
                    vals = [float(v) for v in out.values()]
                else:
                    vals = [float(out)]
                if not all(np.isfinite(v) for v in vals):
                    print(f"  Warning: new reward returned non-finite values; keeping current version")
                    return False

            elif fn_name == "summarize":
                fake_step = (obs, action, obs, {"total": 0.0, "dummy": 0.0}, False)
                out = fn([fake_step, fake_step])
                if not isinstance(out, str):
                    print(f"  Warning: new summarize did not return a string; keeping current version")
                    return False

            return True
        except Exception as e:
            print(f"  Warning: {fn_name} smoke test raised {type(e).__name__}: {e}; keeping current version")
            return False

    def _hot_swap(self, code: str, fn_name: str, rnd: int) -> bool:
        """Returns True if the swap actually went through."""
        code = code.removeprefix("```python").removesuffix("```").strip()
        try:
            fn = _exec_fn(code, fn_name)
        except Exception as e:
            print(f"  Warning: {fn_name} compile failed ({e}); keeping current version")
            return False

        # smoke test — refuse to install something that errors or returns
        # non-finite values on a sample input, since the resulting NaNs poison
        # the whole training-history block fed back into the next reflection.
        if not self._smoke_test(fn, fn_name):
            return False

        if fn_name == "reward":
            self.composite.r_fixed = fn
            self.reward_code       = code
            stem = f"reward_round{rnd}"
            if self.verbose:
                print("  ✓ r_fixed updated")
        elif fn_name == "summarize":
            self.sampler.semantic_fn = fn
            self.semantic_code       = code
            stem = f"semantic_round{rnd}"
            if self.verbose:
                print("  ✓ summarize updated")

        # write a round-versioned snapshot so the full edit history is inspectable
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / f"{stem}.py").write_text(code)
        return True
