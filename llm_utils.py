import hashlib
import math as _math
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from dotenv import load_dotenv

load_dotenv()

PROMPT_DIR = Path(__file__).parent / "prompts"
CACHE_DIR = Path(".llm_cache")


def describe_env(env: gym.Env, task_description: Optional[str] = None) -> Dict[str, Any]:
    """Return metadata describing a Gym environment."""
    spec = getattr(env, "spec", None)
    env_id = spec.id if spec is not None else env.unwrapped.__class__.__name__ # CartPole-v1, LunarLander-v2, HalfCheetah-v4, etc

    def _space_desc(space) -> Dict[str, Any]:
        """Describe the observation or action space."""
        info: Dict[str, Any] = {"type": type(space).__name__} # Box, Discrete, etc
        if hasattr(space, "shape") and space.shape is not None:
            info["shape"] = tuple(space.shape) # Dimension of space
        if hasattr(space, "low") and hasattr(space, "high"):
            # Bounds of each dimension for continuous spaces
            try:
                info["low"] = np.asarray(space.low).tolist()
                info["high"] = np.asarray(space.high).tolist()
            except (TypeError, ValueError):
                pass
        if hasattr(space, "n"):
            info["n"] = int(space.n) # Dimension for discrete spaces
        if hasattr(space, "dtype"):
            info["dtype"] = str(space.dtype) # Data type
        return info

    docstring = (env.unwrapped.__class__.__doc__ or "").strip() # Get docstring of environment class

    return {
        "env_id": env_id,
        "observation_space": _space_desc(env.observation_space),
        "action_space": _space_desc(env.action_space),
        "docstring": docstring,
        "task": task_description,
    }


def _load_prompt(name: str) -> str:
    """Load text prompt from PROMPT_DIR."""
    return (PROMPT_DIR / f"{name}.md").read_text()


def _hash_task(env_id: str, task: str) -> str:
    """Generate a short hex cache ID for an environment-task pair."""
    return hashlib.sha1(f"{env_id}::{task}".encode()).hexdigest()[:12]


def _exec_fn(code: str, fn_name: str) -> Callable:
    """Execute generated code and return the requested function."""
    namespace: Dict[str, Any] = {"np": np, "math": _math} # Give generated code access to libraries
    exec(code, namespace) # Run generated code inside this namespace
    fn = namespace.get(fn_name) # Get generated function from the namespace by name
    if not callable(fn):
        raise RuntimeError(f"Generated code did not define function `{fn_name}`")
    return fn


def _generate_code(
    prompt_name: str,
    fn_name: str,
    env_desc: Dict[str, Any],
    task: str,
    model: str,
) -> Tuple[str, Callable]:
    # Create a cache filename for the environment, task, and prompt
    safe_id = env_desc["env_id"].replace("/", "_")
    task_hash = _hash_task(env_desc["env_id"], task)
    cache_path = CACHE_DIR / f"{safe_id}_{task_hash}_{prompt_name}.py"

    # Return the executable generated code if the result is cached
    if cache_path.exists():
        code = cache_path.read_text()
        return code, _exec_fn(code, fn_name)

    # Otherwise, generate code using the LLM
    from openai import OpenAI
    client = OpenAI()

    # First format the prompt by replacing placeholder values in the prompt using env_desc
    prompt = _load_prompt(prompt_name).format(
        env_id=env_desc["env_id"],
        obs_space=env_desc["observation_space"],
        act_space=env_desc["action_space"],
        docstring=env_desc["docstring"] or "(no docstring available)",
        task=task,
    )

    # Send the formatted prompt to the LLM and generate a response
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    code = resp.choices[0].message.content # Extract the model's generated text
    cache_path.parent.mkdir(parents=True, exist_ok=True) # Create cache directory if needed
    cache_path.write_text(code) # Save generated code to the cache file

    return code, _exec_fn(code, fn_name)


def generate_reward_fn(
    env: gym.Env,
    task_description: str,
    model: str = "gpt-4o-mini",
) -> Tuple[str, Callable]:
    env_desc = describe_env(env, task_description)
    return _generate_code("reward", "reward", env_desc, task_description, model)


def generate_semantic_fn(
    env: gym.Env,
    task_description: str,
    model: str = "gpt-4o-mini",
) -> Tuple[str, Callable]:
    env_desc = describe_env(env, task_description)
    return _generate_code("semantic", "summarize", env_desc, task_description, model)


def compare_trajectories(
    traj_a,
    traj_b,
    semantic_fn: Callable,
    task_description: str,
    model: str = "gpt-4o-mini",
) -> Tuple[int, str]:
    """Ask the LLM which trajectory better satisfies the task.

    Returns (label, explanation) where label is 1 if A preferred, 0 if B preferred.
    """
    import json

    # Get trajectory summaries from the semantic layer
    summary_a = str(semantic_fn(traj_a))
    summary_b = str(semantic_fn(traj_b))

    # Format the prompt by replacing placeholder values in the prompt using the summaries
    prompt = _load_prompt("compare").format(
        task=task_description,
        summary_a=summary_a,
        summary_b=summary_b,
    )

    # Send the formatted prompt to the LLM and generate a response
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or ""

    # Parse result, get preference label and explanation
    parsed = json.loads(raw)
    preference = str(parsed.get("preference", "")).strip().upper()
    explanation = str(parsed.get("explanation", "")).strip()

    if preference.startswith("A"):
        return 1, explanation
    if preference.startswith("B"):
        return 0, explanation
    raise ValueError(f"Unexpected LLM comparison response: {raw!r}")