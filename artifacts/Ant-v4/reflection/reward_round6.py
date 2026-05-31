def reward(obs, action, next_obs):
    forward_speed = float(next_obs[13])  # Reward for forward speed
    height = float(obs[0])                # Current height of the torso
    action_cost = -0.1 * float(np.sum(action ** 2))  # Increased penalty on control effort
    lateral_drift_penalty = -1.0 * abs(float(obs[14]))  # Stronger penalty for lateral movement

    # Healthy height range
    healthy_height_min = 0.2
    healthy_height_max = 1.0
    height_penalty = 0.0
    if height < healthy_height_min:
        height_penalty = -5.0 * (healthy_height_min - height)  # Strong penalty for being too low
    elif height > healthy_height_max:
        height_penalty = -5.0 * (height - healthy_height_max)  # Strong penalty for being too high

    # Normalize components to ensure they are comparable
    temp_height = 1.0
    temp_action = 1.0
    temp_lateral = 1.0
    temp_speed = 1.0

    normalized_forward_speed = np.clip(np.exp(-np.clip(forward_speed / temp_speed, -20.0, 20.0)), 0, 1)
    normalized_height_penalty = np.clip(np.exp(-np.clip(height_penalty / temp_height, -20.0, 20.0)), 0, 1)
    normalized_action_cost = np.clip(np.exp(-np.clip(action_cost / temp_action, -20.0, 20.0)), 0, 1)
    normalized_lateral_drift_penalty = np.clip(np.exp(-np.clip(lateral_drift_penalty / temp_lateral, -20.0, 20.0)), 0, 1)

    total = normalized_forward_speed + normalized_height_penalty + normalized_action_cost + normalized_lateral_drift_penalty
    return {
        "total": total,
        "forward_speed": normalized_forward_speed,
        "height_penalty": normalized_height_penalty,
        "action_cost": normalized_action_cost,
        "lateral_drift_penalty": normalized_lateral_drift_penalty
    }