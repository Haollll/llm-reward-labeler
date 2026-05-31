def reward(obs, action, next_obs):
    height = obs[0]
    torso_angle = obs[1]
    forward_velocity = next_obs[8]

    # Reward for moving forward
    progress = forward_velocity

    # Survival bonus for staying upright and healthy
    height_bonus = 1.0 if 0.7 < height < 2.0 else -2.0
    angle_bonus = 1.0 if -0.2 < torso_angle < 0.2 else -2.0
    survival_bonus = height_bonus + angle_bonus

    # Penalty for large control torques (energy)
    action_cost = -0.1 * float(np.sum(action ** 2))

    # Normalize components to avoid dominance
    temp_progress = 1.0
    temp_survival = 1.0
    temp_action_cost = 1.0

    # Normalize rewards
    normalized_progress = np.clip(np.exp(forward_velocity / temp_progress), 0, 1)
    normalized_survival = np.clip(np.exp(survival_bonus / temp_survival), 0, 1)
    normalized_action_cost = np.clip(np.exp(action_cost / temp_action_cost), 0, 1)

    # Total reward calculation
    total = normalized_progress + normalized_survival + normalized_action_cost

    return {
        'total': total,
        'progress': normalized_progress,
        'survival_bonus': normalized_survival,
        'action_cost': normalized_action_cost
    }