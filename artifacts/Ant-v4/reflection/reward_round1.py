def reward(obs, action, next_obs):
    forward_speed = float(next_obs[13])  # Reward for forward speed
    height = float(obs[0])                # Current height of the torso
    action_cost = -0.1 * float(np.sum(action ** 2))  # Penalty on control effort
    lateral_drift_penalty = -0.5 * abs(float(obs[14]))  # Penalize lateral movement

    # Healthy height range (example values, adjust as necessary)
    healthy_height_min = 0.2
    healthy_height_max = 1.0
    height_penalty = 0.0
    if height < healthy_height_min:
        height_penalty = -2.0  # Strong penalty for being too low
    elif height > healthy_height_max:
        height_penalty = -2.0  # Strong penalty for being too high

    # Normalize forward speed to encourage faster movement
    temp_forward_speed = 1.0
    normalized_forward_speed = np.clip(np.exp(-forward_speed / temp_forward_speed), 0, 1)

    total = normalized_forward_speed + height_penalty + action_cost + lateral_drift_penalty
    return {
        'total': total,
        'forward_speed': normalized_forward_speed,
        'height_penalty': height_penalty,
        'action_cost': action_cost,
        'lateral_drift_penalty': lateral_drift_penalty
    }