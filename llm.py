"""LLM interface for v2.

Three roles, all backed by OpenAI chat completions:
  * `generate_reward_fn` / `generate_semantic_fn` — write the initial coded
    reward function and trajectory summarizer (with compile/validate + retry +
    on-disk caching so a fixed (env, task) pair is only generated once).
  * `compare_trajectories` — label a preference over two full trajectories.
  * `reflect` — Phase-I reflection that proposes new reward / summarizer code
    from the round-over-round training feedback.

Generated code runs in a restricted namespace exposing only `np` and `math`.
"""

import hashlib
import json
import math as _math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from dotenv import load_dotenv

load_dotenv()

PROMPT_DIR = Path(__file__).parent / "prompts"
CACHE_DIR = Path(".llm_cache")


# ─────────────────────────────────────────────────────────────
# Env description + prompt helpers
# ─────────────────────────────────────────────────────────────

def describe_env(env: gym.Env, task_description: Optional[str] = None) -> Dict[str, Any]:
    spec = getattr(env, "spec", None)
    env_id = spec.id if spec is not None else env.unwrapped.__class__.__name__

    def _space_desc(space) -> Dict[str, Any]:
        info: Dict[str, Any] = {"type": type(space).__name__}
        if hasattr(space, "shape") and space.shape is not None:
            info["shape"] = tuple(space.shape)
        if hasattr(space, "low") and hasattr(space, "high"):
            try:
                info["low"] = np.asarray(space.low).tolist()
                info["high"] = np.asarray(space.high).tolist()
            except (TypeError, ValueError):
                pass
        if hasattr(space, "n"):
            info["n"] = int(space.n)
        if hasattr(space, "dtype"):
            info["dtype"] = str(space.dtype)
        return info

    docstring = (env.unwrapped.__class__.__doc__ or "").strip()
    return {
        "env_id": env_id,
        "observation_space": _space_desc(env.observation_space),
        "action_space": _space_desc(env.action_space),
        "docstring": docstring,
        "task": task_description,
    }


def _load_prompt(name: str) -> str:
    return (PROMPT_DIR / f"{name}.md").read_text()


def _hash_task(env_id: str, task: str) -> str:
    return hashlib.sha1(f"{env_id}::{task}".encode()).hexdigest()[:12]


def cache_key(env: gym.Env, task: str) -> str:
    env_id = describe_env(env)["env_id"]
    safe_id = env_id.replace("/", "_")
    return f"{safe_id}_{_hash_task(env_id, task)}"


# ─────────────────────────────────────────────────────────────
# Code compilation
# ─────────────────────────────────────────────────────────────

def exec_fn(code: str, fn_name: str) -> Callable:
    """Compile generated source and return the named function. Only `np` and
    `math` are exposed to the generated code."""
    namespace: Dict[str, Any] = {"np": np, "math": _math}
    exec(code, namespace)
    fn = namespace.get(fn_name)
    if not callable(fn):
        raise RuntimeError(f"Generated code did not define function `{fn_name}`")
    return fn


def strip_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[1] if "\n" in code else code[3:]
        code = code.removeprefix("python").lstrip("\n")
    return code.removesuffix("```").strip()


def _compile_and_validate(code: str, fn_name: str, validate):
    code = strip_fences(code)
    try:
        fn = exec_fn(code, fn_name)
    except Exception as e:
        return None, f"compile error: {type(e).__name__}: {e}"
    if validate is not None:
        try:
            ok = validate(fn)
        except Exception as e:
            return None, f"runtime error on sample input: {type(e).__name__}: {e}"
        if not ok:
            return None, "validation failed (bad output type / non-finite values)"
    return fn, None


def _client():
    from openai import OpenAI
    return OpenAI()


def _generate_code(
    prompt_name: str,
    fn_name: str,
    env_desc: Dict[str, Any],
    task: str,
    model: str,
    validate=None,
    max_attempts: int = 4,
) -> Tuple[str, Callable]:
    safe_id = env_desc["env_id"].replace("/", "_")
    task_hash = _hash_task(env_desc["env_id"], task)
    cache_path = CACHE_DIR / f"{safe_id}_{task_hash}_{prompt_name}.py"

    if cache_path.exists():
        code = strip_fences(cache_path.read_text())
        fn, err = _compile_and_validate(code, fn_name, validate)
        if fn is not None:
            return code, fn
        print(f"  ⚠ cached {prompt_name} fn invalid ({err}); regenerating")

    base_prompt = _load_prompt(prompt_name).format(
        env_id=env_desc["env_id"],
        obs_space=env_desc["observation_space"],
        act_space=env_desc["action_space"],
        docstring=env_desc["docstring"] or "(no docstring available)",
        task=task,
    )

    client = _client()
    last_err = "unknown"
    for attempt in range(max_attempts):
        prompt = base_prompt
        if attempt > 0:
            prompt += (
                f"\n\n## Your previous attempt failed\n"
                f"It raised: {last_err}\n"
                f"Return a corrected, complete `{fn_name}` definition that runs "
                f"without error on the given spaces. Output ONLY the function "
                f"definition, no markdown fences."
            )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2 + 0.2 * attempt,
        )
        code = strip_fences(resp.choices[0].message.content or "")
        fn, err = _compile_and_validate(code, fn_name, validate)
        if fn is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(code)
            return code, fn
        last_err = err
        print(f"  ⚠ generated {prompt_name} fn invalid (attempt {attempt + 1}/"
              f"{max_attempts}): {err}")

    raise RuntimeError(
        f"Could not generate a valid `{fn_name}` for {env_desc['env_id']} after "
        f"{max_attempts} attempts. Last error: {last_err}"
    )


# ─────────────────────────────────────────────────────────────
# Public generators
# ─────────────────────────────────────────────────────────────

def generate_reward_fn(env: gym.Env, task: str, model: str = "gpt-4o-mini") -> Tuple[str, Callable]:
    env_desc = describe_env(env, task)

    def _validate(fn) -> bool:
        obs = env.observation_space.sample()
        action = env.action_space.sample()
        out = fn(obs, action, obs)
        vals = out.values() if isinstance(out, dict) else [out]
        return all(np.isfinite(float(v)) for v in vals)

    return _generate_code("reward", "reward", env_desc, task, model, validate=_validate)


def generate_semantic_fn(env: gym.Env, task: str, model: str = "gpt-4o-mini") -> Tuple[str, Callable]:
    env_desc = describe_env(env, task)

    def _validate(fn) -> bool:
        obs = env.observation_space.sample()
        action = env.action_space.sample()
        step = (obs, action, obs, {"total": 0.0, "component_a": 0.1}, False)
        out = fn([step, step])
        return isinstance(out, str) and len(out) > 0

    return _generate_code("semantic", "summarize", env_desc, task, model, validate=_validate)


# ─────────────────────────────────────────────────────────────
# Preference comparison
# ─────────────────────────────────────────────────────────────

def compare_trajectories(
    traj_a,
    traj_b,
    semantic_fn: Callable,
    task: str,
    model: str = "gpt-4o-mini",
) -> Tuple[int, str]:
    """Ask the LLM which full trajectory better achieves the task.

    Returns (label, explanation): label is 1 if A preferred, 0 if B preferred.
    """
    summary_a = str(semantic_fn(traj_a))
    summary_b = str(semantic_fn(traj_b))
    prompt = _load_prompt("compare").format(task=task, summary_a=summary_a, summary_b=summary_b)

    client = _client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or ""
    parsed = json.loads(raw)
    preference = str(parsed.get("preference", "")).strip().upper()
    explanation = str(parsed.get("explanation", "")).strip()
    if preference.startswith("A"):
        return 1, explanation
    if preference.startswith("B"):
        return 0, explanation
    raise ValueError(f"Unexpected LLM comparison response: {raw!r}")


# ─────────────────────────────────────────────────────────────
# Phase-I reflection
# ─────────────────────────────────────────────────────────────

def _format_series(name: str, vals: List[float], fmt: str) -> str:
    quoted = ", ".join(f"'{v:{fmt}}'" for v in vals)
    return (f"{name}: [{quoted}], Max: {max(vals):{fmt}}, "
            f"Mean: {sum(vals)/len(vals):{fmt}}, Min: {min(vals):{fmt}}")


def format_training_history(snapshots: List[dict]) -> str:
    """snapshots: list of {component_means: {..}, episode_length, episode_env_reward}."""
    if not snapshots:
        return "No training data yet."
    lines: List[str] = []
    all_comps: set = set()
    for s in snapshots:
        all_comps.update(s["component_means"].keys())
    for comp in sorted(all_comps):
        vals = [s["component_means"].get(comp, 0.0) for s in snapshots]
        lines.append(_format_series(comp, vals, ".3f"))
    lines.append(_format_series("episode_length", [s["episode_length"] for s in snapshots], ".1f"))
    lines.append(_format_series("episode_env_reward", [s["episode_env_reward"] for s in snapshots], ".2f"))
    return "\n".join(lines)


def reflect(
    task: str,
    reward_code: str,
    semantic_code: str,
    snapshots: List[dict],
    model: str = "gpt-4o",
) -> dict:
    """Run one reflection call. Returns the parsed JSON
    {analysis, reward_code, semantic_code, reasoning}."""
    prompt = _load_prompt("reflection").format(
        task=task,
        reward_code=reward_code,
        semantic_code=semantic_code,
        training_summary=format_training_history(snapshots),
    )
    client = _client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return json.loads(resp.choices[0].message.content)
