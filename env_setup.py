import gymnasium as gym
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple

if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

# ─────────────────────────────────────────
# HalfCheetah Observation Space (17 dims)
# ─────────────────────────────────────────
# obs[0]     : z position (height)
# obs[1]     : body pitch angle
# obs[2:8]   : joint angles (6 total)
# obs[8]     : x velocity (forward speed) ← most important
# obs[9]     : z velocity (vertical velocity)
# obs[10:17] : joint angular velocities (7 total)
#
# Action space (6 dims): joint torques in [-1, 1]

ENV_CONTEXT = """
Task: Make the HalfCheetah robot run to the right as fast as possible.

Observation space (17 dimensions):
- Body height and pitch angle
- 6 joint angles (hip, knee, ankle of front/back legs)
- Forward velocity (x-direction) ← most critical metric
- Vertical velocity
- Joint angular velocities

Action space (6 dimensions): torques applied to each joint, range [-1, 1]

Definition of good behavior:
1. High forward velocity (larger x velocity is better)
2. Smooth actions (small changes between consecutive actions)
3. Reasonable energy usage (avoid excessively large torques)
4. Stable body posture (small variation in height and pitch)

Definition of bad behavior:
1. Slow or backward movement (negative x velocity)
2. Highly oscillatory actions (unstable control)
3. Excessive energy usage (all joints using max torque)
"""


@dataclass
class TrajectoryFeatures:
    mean_forward_velocity: float    # average forward velocity
    total_displacement: float       # total displacement (estimated)
    action_smoothness: float        # action smoothness (lower = smoother)
    mean_control_effort: float      # average control effort
    height_stability: float         # body height stability (std)
    pitch_stability: float          # pitch stability (std)
    n_steps: int                    # number of steps


def extract_features(trajectory: List[Tuple]) -> TrajectoryFeatures:
    obs_list     = np.array([t[0] for t in trajectory])
    action_list  = np.array([t[1] for t in trajectory])

    # Forward velocity (obs index 8)
    forward_velocities = obs_list[:, 8]
    mean_fwd_vel = float(np.mean(forward_velocities))

    # Estimated total displacement (velocity × dt, HalfCheetah dt=0.05)
    total_disp = float(np.sum(forward_velocities) * 0.05)

    # Action smoothness: mean L2 difference between consecutive actions
    if len(action_list) > 1:
        diffs = np.diff(action_list, axis=0)
        smoothness = float(np.mean(np.linalg.norm(diffs, axis=1)))
    else:
        smoothness = 0.0

    # Control effort: mean L2 norm of actions
    control_effort = float(np.mean(np.linalg.norm(action_list, axis=1)))

    # Body stability
    height_std = float(np.std(obs_list[:, 0]))   # z position
    pitch_std  = float(np.std(obs_list[:, 1]))   # pitch angle

    return TrajectoryFeatures(
        mean_forward_velocity=mean_fwd_vel,
        total_displacement=total_disp,
        action_smoothness=smoothness,
        mean_control_effort=control_effort,
        height_stability=height_std,
        pitch_stability=pitch_std,
        n_steps=len(trajectory),
    )


def features_to_text(features: TrajectoryFeatures) -> str:
    direction = "forward" if features.mean_forward_velocity > 0 else "backward (negative, poor)"

    smoothness_desc = (
        "very smooth" if features.action_smoothness < 0.1 else
        "smooth"      if features.action_smoothness < 0.3 else
        "slightly oscillatory" if features.action_smoothness < 0.6 else
        "highly oscillatory (unstable control)"
    )

    effort_desc = (
        "energy-efficient" if features.mean_control_effort < 0.3 else
        "moderate"         if features.mean_control_effort < 0.7 else
        "high energy consumption"
    )

    return f"""
Trajectory Summary ({features.n_steps} steps):
- Mean forward velocity: {features.mean_forward_velocity:.3f} m/s ({direction})
- Estimated total displacement: {features.total_displacement:.3f} m
- Action smoothness: {smoothness_desc} (value {features.action_smoothness:.3f})
- Control effort: {effort_desc} (value {features.mean_control_effort:.3f})
- Height stability (std): {features.height_stability:.4f}
- Pitch stability (std): {features.pitch_stability:.4f}
""".strip()


def collect_trajectory(env, policy_fn, max_steps: int = 100) -> List[Tuple]:

    trajectory = []
    obs, _ = env.reset()

    for _ in range(max_steps):
        action = policy_fn(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        trajectory.append((obs, action, next_obs, reward, done))
        obs = next_obs
        if done:
            break

    return trajectory


if __name__ == "__main__":
    env = gym.make("HalfCheetah-v5")

    # Test feature extraction with a random policy
    random_policy = lambda obs: env.action_space.sample()
    traj = collect_trajectory(env, random_policy, max_steps=50)

    features = extract_features(traj)
    print("=== Feature Extraction Test ===")
    print(features_to_text(features))
    env.close()