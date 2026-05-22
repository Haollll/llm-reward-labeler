from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from openai import OpenAI
from llm_utils import _exec_fn, _load_prompt, CACHE_DIR

client = OpenAI()


# ── data ─────────────────────────────────────────────────────

@dataclass
class EvalStats:
    rnd:        int
    reward:     float
    loss:       float
    acc:        float
    n_labels:   int
    components: Dict[str, float] = field(default_factory=dict)


# ── prompt ───────────────────────────────────────────────────

def _build_prompt(status: str, reward_code: str, semantic_code: str, task: str) -> str:
    return _load_prompt("reflection").format(
        status        = status,
        reward_code   = reward_code,
        semantic_code = semantic_code,
        task          = task,
    )


def _status(history: List[EvalStats]) -> str:
    if not history:
        return "No data yet."
    s = history[-1]

    lines = [
        f"Round {s.rnd} | reward {s.reward:.1f} | loss {s.loss:.4f} "
        f"| acc {s.acc:.1%} | labels {s.n_labels}",
    ]

    # component breakdown — helps LLM diagnose which part of r_fixed is wrong
    if s.components:
        non_total = {k: v for k, v in s.components.items() if k != "total"}
        if non_total:
            breakdown = " | ".join(f"{k} {v:+.2f}" for k, v in non_total.items())
            lines.append(f"Components: {breakdown}")

    if len(history) >= 2:
        delta = s.reward - history[-2].reward
        lines.append(f"Δ reward {'+' if delta >= 0 else ''}{delta:.1f}")

    if len(history) >= 2:
        recent = [h.reward for h in history[-5:]]
        delta  = recent[-1] - recent[0]
        trend  = "improving" if delta > 50 else "declining" if delta < -50 else "plateauing"
        lines.append(f"Trend: {trend}")

    return " | ".join(lines)


# ── engine ───────────────────────────────────────────────────

class ReflectionEngine:
    """
    Thin coordinator: tracks stats, decides when to reflect, applies result.
    All LLM logic is in _reflect().
    All apply logic is in _apply().
    """

    def __init__(
        self,
        task: str,
        reward_code: str,
        semantic_code: str,
        composite_reward,
        sampler,
        reflect_every: int        = 3,
        acc_threshold: float      = 0.70,
        plateau_rounds: int       = 3,
        llm_model: str            = "gpt-4o",
        verbose: bool             = True,
    ):
        self.task             = task
        self.reward_code      = reward_code
        self.semantic_code    = semantic_code
        self.composite        = composite_reward
        self.sampler          = sampler
        self.reflect_every    = reflect_every
        self.acc_threshold    = acc_threshold
        self.plateau_rounds   = plateau_rounds
        self.llm_model        = llm_model
        self.verbose          = verbose

        self._history: List[EvalStats] = []
        self._plateau: int             = 0
        self._n_evals: int             = 0
        self.log:      list            = []

    # ── public ───────────────────────────────────────────────

    def step(
        self,
        rnd: int,
        reward: float,
        loss: float,
        acc: float,
        n_labels: int,
    ) -> Optional[dict]:
        """Record stats and reflect if any trigger fires."""
        components = dict(getattr(self.composite, "last_components", {}))
        self._history.append(EvalStats(rnd, reward, loss, acc, n_labels, components))
        self._n_evals += 1

        if len(self._history) >= 2:
            trend = _status(self._history).split("Trend: ")[-1]
            self._plateau = self._plateau + 1 if trend == "plateauing" else 0

        if self._should_reflect(acc):
            return self._reflect()
        return None

    # ── private: trigger ─────────────────────────────────────

    def _should_reflect(self, acc: float) -> bool:
        if not self._history:
            return False
        if acc < self.acc_threshold:
            return True
        if self._plateau >= self.plateau_rounds:
            return True
        if self._n_evals % self.reflect_every == 0:
            return True
        return False

    # ── private: LLM call ────────────────────────────────────

    def _reflect(self) -> dict:
        if self.verbose:
            print("\n  [Reflection] Calling LLM...")

        prompt = _build_prompt(
            status        = _status(self._history),
            reward_code   = self.reward_code,
            semantic_code = self.semantic_code,
            task          = self.task,
        )
        resp = client.chat.completions.create(
            model           = self.llm_model,
            messages        = [{"role": "user", "content": prompt}],
            response_format = {"type": "json_object"},
            temperature     = 0.3,
        )
        result = json.loads(resp.choices[0].message.content)

        if self.verbose:
            print(f"  → {result.get('option','?')} | {result.get('diagnosis','')}")

        self._apply(result)
        self._cache(result)
        self.log.append(result)
        self._plateau = 0
        return result

    # ── private: apply ───────────────────────────────────────

    def _apply(self, result: dict) -> None:
        option   = result.get("option", "")
        code     = result.get("code")
        fn_name  = result.get("fn_name")
        guidance = result.get("guidance")

        if option in ("A", "B") and code and fn_name:
            self._hot_swap(fn_name, code)
        elif option == "C" and guidance:
            self.sampler.comparison_guidance = guidance
            if self.verbose:
                print("  ✓ guidance updated")

    def _hot_swap(self, fn_name: str, code: str) -> None:
        code = code.removeprefix("```python").removesuffix("```").strip()
        try:
            fn = _exec_fn(code, fn_name)
        except Exception as e:
            print(f"  Warning: compile failed ({e}), keeping old version")
            return
        if fn_name == "reward":
            self.composite.r_fixed = fn
            self.reward_code       = code
            if self.verbose:
                print("  ✓ r_fixed updated")
        elif fn_name == "summarize":
            self.sampler.semantic_fn = fn
            self.semantic_code       = code
            if self.verbose:
                print("  ✓ semantic layer updated")

    # ── private: cache ───────────────────────────────────────

    def _cache(self, result: dict) -> None:
        if result.get("option") not in ("A", "B"):
            return
        code    = result.get("code")
        fn_name = result.get("fn_name")
        if not code or not fn_name:
            return
        name = "reward_reflected" if fn_name == "reward" else "semantic_reflected"
        path = CACHE_DIR / f"{name}.py"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(code)