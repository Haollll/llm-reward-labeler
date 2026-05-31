def reward(obs, action, next_obs) -> dict:
    temp_action = 0.5
    temp_stability = 0.1
    temp_speed = 0.1

    forward_speed = next_obs[8]  # Reward for forward speed
    action_cost = -np.clip(np.exp(-np.sum(action ** 2) / temp_action), 0, 1)  # Penalty for high torque usage
    stability_penalty = -np.clip(np.exp((np.abs(obs[1]) + np.sum(np.abs(obs[2:8]))) / temp_stability), 0, 1)  # Penalty for body angle and joint angles

    total = forward_speed + action_cost + stability_penalty
    return {
        'total': total,
        'forward_speed': forward_speed,
        'action_cost': action_cost,
        'stability_penalty': stability_penalty
    }