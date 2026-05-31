def reward(obs, action, next_obs):
    forward_velocity = next_obs[3]  # Reward for forward speed
    action_cost = -0.1 * float(np.sum(action ** 2))  # Penalty for high torque usage
    smoothness_penalty = -0.05 * (np.abs(obs[6]) + np.abs(obs[7]))  # Penalty for joint velocity

    # Normalize components to ensure balanced contribution
    temp_action_cost = 10.0
    temp_smoothness = 5.0
    normalized_forward_velocity = np.clip(forward_velocity, 0, None)
    normalized_action_cost = np.exp(-np.clip(action_cost / temp_action_cost, -20.0, 20.0))
    normalized_smoothness_penalty = np.exp(-np.clip(smoothness_penalty / temp_smoothness, -20.0, 20.0))

    total = normalized_forward_velocity + normalized_action_cost + normalized_smoothness_penalty
    return {
        'total': total,
        'forward_velocity': normalized_forward_velocity,
        'action_cost': normalized_action_cost,
        'smoothness_penalty': normalized_smoothness_penalty
    }