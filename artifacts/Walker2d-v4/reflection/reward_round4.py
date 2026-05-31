def reward(obs, action, next_obs) -> dict:
    height = obs[0]
    torso_angle = obs[1]
    forward_velocity = next_obs[8]

    # Reward for moving forward
    progress = forward_velocity

    # Survival bonus for staying upright and healthy
    height_bonus = 1.0 if 0.7 < height < 2.0 else -1.0
    angle_bonus = 1.0 if -0.2 < torso_angle < 0.2 else -1.0
    survival_bonus = height_bonus + angle_bonus

    # Penalty for large control torques (energy)
    action_cost_temp = 5.0  # Temperature for action cost
    action_cost = -np.clip(np.exp(-np.sum(action ** 2) / action_cost_temp), 0, 1)

    # Normalize progress and survival bonuses to a comparable range
    progress_temp = 10.0  # Temperature for progress
    normalized_progress = np.clip(np.exp(forward_velocity / progress_temp), 0, 1)
    normalized_survival_bonus = np.clip(survival_bonus, -2, 2)  # Scale survival bonus to [-2, 2]

    # Total reward calculation
    total = normalized_progress + normalized_survival_bonus + action_cost

    return {
        "total": total,
        "progress": normalized_progress,
        "survival_bonus": normalized_survival_bonus,
        "action_cost": action_cost
    }