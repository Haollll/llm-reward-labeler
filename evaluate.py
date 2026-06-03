"""Evaluate a trained v2 pipeline policy on the TRUE env reward and gather the
data needed for the result plots.

Writes per env:
  * artifacts/<env>/eval.json      — pipeline policy: 100-episode return/length/success
  * artifacts/<env>/baseline/metrics.json — RL-Zoo3 baseline (via baseline.py)
  * artifacts/<env>/bt_vs_env.json — single-episode per-step BT reward vs true env
                                     reward (rolled out with the baseline policy)

    python evaluate.py --env HalfCheetah-v4
"""

import argparse
import json
import os
from pathlib import Path

import helper  # noqa: F401  (env-warning suppression)
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from env_utils import eval_with_components
from helper import load_task, task_for_env, success_fn_for_env
from reward_model import BTRewardModel
import baseline as baseline_mod
from paths import policy_dir, reward_model_dir, eval_metrics_path, bt_vs_env_path, env_dir


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a v2 pipeline policy vs the baseline")
    p.add_argument("--env", default="HalfCheetah-v4")
    p.add_argument("--task", default=None)
    p.add_argument("--artifact-dir", default="artifacts")
    p.add_argument("--baseline-dir", default="baselines")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--render", action="store_true",
                   help="Visually render the trained policy instead of the headless eval")
    p.add_argument("--render-episodes", type=int, default=1,
                   help="Number of episodes to render when --render is set")
    p.add_argument("--zoom", type=float, default=1.0,
                   help="Camera distance multiplier for --render (>1 zooms out, <1 zooms in)")
    p.add_argument("--compare", action="store_true",
                   help="With --render, also roll out the baseline and write a side-by-side GIF")
    return p.parse_args()


def _load_obs_normalizer(env_id, policy_path):
    """If the policy was trained with VecNormalize, reload the saved obs stats so
    we can normalize observations the same way at eval time. Returns a callable
    obs->obs (identity if no stats were saved)."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    stats = Path(policy_path).parent / "vecnormalize.pkl"
    if not stats.exists():
        return lambda o: o
    venv = DummyVecEnv([lambda: gym.make(env_id)])
    vn = VecNormalize.load(str(stats), venv)
    vn.training = False
    vn.norm_reward = False
    return lambda o: vn.normalize_obs(o)


def evaluate_pipeline(env_id, policy_path, episodes, sfn):
    env = gym.make(env_id)
    policy = PPO.load(str(policy_path), device="cpu")
    norm = _load_obs_normalizer(env_id, policy_path)
    data = eval_with_components(env, lambda o: policy.predict(norm(o), deterministic=True)[0],
                               reward_fn=None, n_episodes=episodes)
    env.close()
    ep_rewards = [float(x) for x in data["episode_env_rewards"]]
    ep_lengths = [int(x) for x in data["episode_lengths"]]
    succ = (float(np.mean([1.0 if sfn(l, r) else 0.0
                           for l, r in zip(ep_lengths, ep_rewards)]))
            if sfn is not None else None)
    return {
        "env_id": env_id,
        "n_episodes": episodes,
        "episode_env_rewards": ep_rewards,
        "episode_lengths": ep_lengths,
        "mean_return": float(np.mean(ep_rewards)),
        "mean_length": float(np.mean(ep_lengths)),
        "success_rate": succ,
    }


def _setup_scene_camera(env, zoom):
    """Switch the render env to a zoomable, tracking FREE camera and reveal the
    wider scene. Returns the cam object to re-center each frame, or None when
    zoom == 1.0 (in which case the env keeps its tight built-in tracking cam).

    The env's default camera (camera_id 0) is a FIXED tracking cam that ignores
    distance, so we select the FREE cam (camera_id -1), scale its distance by
    `model.stat.extent * zoom`, kill the distance haze that greys the floor when
    pulled back, and enlarge the finite checker plane (+ its texture repeat) so a
    fast agent never runs off its edge."""
    if zoom == 1.0:
        return None
    import mujoco
    model = env.unwrapped.model

    FLOOR_SCALE = 50.0
    for g in range(model.ngeom):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            model.geom_size[g][0] *= FLOOR_SCALE
            model.geom_size[g][1] *= FLOOR_SCALE
            mat = int(model.geom_matid[g])
            if mat >= 0:
                model.mat_texrepeat[mat][0] *= FLOOR_SCALE
                model.mat_texrepeat[mat][1] *= FLOOR_SCALE

    env.reset()
    env.render()  # lazily creates the viewer
    mr = env.unwrapped.mujoco_renderer
    mr.camera_id = -1
    viewer = mr.viewer
    viewer.scn.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 0
    cam = viewer.cam
    cam.azimuth, cam.elevation = 90.0, -12.0
    cam.distance = float(model.stat.extent) * zoom
    return cam


def _render_episodes(env, policy_fn, episodes, zoom, label=None):
    """Roll out `episodes` full episodes, returning (frames, results). `policy_fn`
    maps a raw obs to an action. Frames track the agent when zoomed; if `label`
    is given it is drawn in the top-left corner of every frame."""
    cam = _setup_scene_camera(env, zoom)

    def _grab():
        if cam is not None:
            torso = env.unwrapped.data.xpos[1]
            cam.lookat[0], cam.lookat[1] = float(torso[0]), float(torso[1])
        frame = env.render()
        return _label_frame(frame, label) if label else frame

    frames, results = [], []
    for ep in range(episodes):
        obs, _ = env.reset()
        ep_reward, ep_len, done = 0.0, 0, False
        frames.append(_grab())
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            ep_len += 1
            done = terminated or truncated
            frames.append(_grab())
        results.append((ep_reward, ep_len))
        tag = f"[{label}] " if label else ""
        print(f"  {tag}episode {ep + 1:>2}/{episodes} | return {ep_reward:10.1f} | length {ep_len:5d}")
    return frames, results


def _label_frame(frame, text):
    """Draw a text label in the top-left corner of an RGB frame."""
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 9 * len(text) + 12, 22], fill=(0, 0, 0))
    draw.text((6, 5), text, fill=(255, 255, 255))
    return np.asarray(img)


def _pipeline_policy_fn(env_id, policy_path):
    """Action function for the trained pipeline policy (our LLM-reward model)."""
    policy = PPO.load(str(policy_path), device="cpu")
    norm = _load_obs_normalizer(env_id, policy_path)
    return lambda obs: policy.predict(norm(obs), deterministic=True)[0]


def _baseline_policy_fn(env_id, baseline_dir):
    """Action function for the RL-Zoo3 PPO baseline, applying its saved obs
    normalization (the baseline was trained under VecNormalize)."""
    model, venv = baseline_mod.load_baseline(env_id, baseline_dir)
    if isinstance(venv, VecNormalize):
        venv.training = False
        venv.norm_reward = False
        norm = venv.normalize_obs
    else:
        norm = lambda o: o
    return lambda obs: model.predict(norm(obs), deterministic=True)[0]


def render_pipeline(env_id, policy_path, episodes, out_path, zoom=1.0):
    """Roll out the trained pipeline policy and save a GIF of it acting.

    A live ("human") window can't be created on this machine's WSL/software-GL
    stack, so we render OFFSCREEN with MuJoCo's OSMesa backend (rgb_array) and
    write the frames to a GIF instead."""
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    import imageio

    env = gym.make(env_id, render_mode="rgb_array")
    fps = int(env.metadata.get("render_fps", 30))
    frames, results = _render_episodes(env, _pipeline_policy_fn(env_id, policy_path),
                                       episodes, zoom)
    env.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=fps)
    print(f"Saved {len(frames)} frames → {out_path}")
    return results


def render_compare(env_id, policy_path, episodes, out_path, baseline_dir, zoom=1.0):
    """Render the pipeline policy and the RL-Zoo3 baseline and write a single GIF
    with the two rollouts side by side (pipeline left, baseline right)."""
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    import imageio

    env_p = gym.make(env_id, render_mode="rgb_array")
    fps = int(env_p.metadata.get("render_fps", 30))
    frames_p, _ = _render_episodes(env_p, _pipeline_policy_fn(env_id, policy_path),
                                   episodes, zoom, label="pipeline")
    env_p.close()

    env_b = gym.make(env_id, render_mode="rgb_array")
    frames_b, _ = _render_episodes(env_b, _baseline_policy_fn(env_id, baseline_dir),
                                   episodes, zoom, label="baseline")
    env_b.close()

    # The two rollouts differ in length; pad the shorter with its last frame so
    # both tracks play out fully, then stack each pair horizontally.
    n = max(len(frames_p), len(frames_b))
    frames_p += [frames_p[-1]] * (n - len(frames_p))
    frames_b += [frames_b[-1]] * (n - len(frames_b))
    sep = np.zeros((frames_p[0].shape[0], 4, 3), dtype=np.uint8)
    combined = [np.concatenate([p, sep, b], axis=1) for p, b in zip(frames_p, frames_b)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), combined, fps=fps)
    print(f"Saved {len(combined)} side-by-side frames → {out_path}")


def bt_vs_env_episode(env_id, artifact_dir, baseline_dir):
    """Roll out the baseline policy for ONE episode on the true env, recording the
    per-step true env reward and the learned BT model's per-step reward, so we can
    see whether the learned reward tracks the true reward."""
    rm = BTRewardModel(gym.make(env_id))
    rm.load(str(reward_model_dir(env_id, artifact_dir)))

    model, venv = baseline_mod.load_baseline(env_id, baseline_dir)
    is_norm = isinstance(venv, VecNormalize)
    obs = venv.reset()

    env_rewards, bt_rewards = [], []
    done = False
    steps = 0
    while not done and steps < 2000:
        action, _ = model.predict(obs, deterministic=True)
        raw_obs = venv.get_original_obs()[0] if is_norm else obs[0]
        obs, reward, dones, _ = venv.step(action)
        env_rewards.append(float(reward[0]))
        bt_rewards.append(float(rm.predict(np.asarray(raw_obs), np.asarray(action[0]))))
        done = bool(dones[0])
        steps += 1
    venv.close()
    return {"env_id": env_id, "env_reward": env_rewards, "bt_reward": bt_rewards}


def main() -> None:
    args = parse_args()
    _ = load_task(args.task or task_for_env(args.env))
    sfn = success_fn_for_env(args.env)

    policy_path = policy_dir(args.env, args.artifact_dir) / "policy.zip"
    if not policy_path.exists():
        raise FileNotFoundError(f"No pipeline policy at {policy_path} (train with train.py)")

    # ── render-only mode ─────────────────────────────────────
    if args.render:
        if args.compare:
            out_path = env_dir(args.env, args.artifact_dir) / "render_compare.gif"
            print(f"Rendering {args.render_episodes} episode(s) of {args.env} "
                  f"(pipeline vs baseline) → {out_path}")
            render_compare(args.env, policy_path, args.render_episodes, out_path,
                           args.baseline_dir, args.zoom)
        else:
            out_path = env_dir(args.env, args.artifact_dir) / "render.gif"
            print(f"Rendering {args.render_episodes} episode(s) of {args.env} → {out_path}")
            render_pipeline(args.env, policy_path, args.render_episodes, out_path, args.zoom)
        return

    # ── pipeline ─────────────────────────────────────────────
    pipe = evaluate_pipeline(args.env, policy_path, args.episodes, sfn)
    ep = eval_metrics_path(args.env, args.artifact_dir)
    ep.parent.mkdir(parents=True, exist_ok=True)
    ep.write_text(json.dumps(pipe, indent=2))
    print(f"Pipeline | {args.episodes} ep | return {pipe['mean_return']:10.1f} | "
          f"length {pipe['mean_length']:6.0f}")

    # ── baseline ─────────────────────────────────────────────
    if not args.no_baseline:
        try:
            base = baseline_mod.evaluate_baseline(
                args.env, n_episodes=args.episodes, baseline_dir=args.baseline_dir,
                artifact_dir=args.artifact_dir, success_fn=sfn)
            print(f"Baseline | {args.episodes} ep | return {base['mean_return']:10.1f} | "
                  f"length {base['mean_length']:6.0f}")
        except Exception as e:
            print(f"[baseline skipped] {type(e).__name__}: {e}")

    # ── per-step BT vs true env reward (single episode) ──────
    try:
        series = bt_vs_env_episode(args.env, args.artifact_dir, args.baseline_dir)
        bp = bt_vs_env_path(args.env, args.artifact_dir)
        bp.write_text(json.dumps(series, indent=2))
        print(f"BT-vs-env series ({len(series['env_reward'])} steps) → {bp}")
    except Exception as e:
        print(f"[bt-vs-env skipped] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
