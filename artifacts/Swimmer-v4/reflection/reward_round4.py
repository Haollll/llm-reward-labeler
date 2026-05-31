def reward(obs, action, next_obs) -> dict:
    forward_velocity = next_obs[3]  # Reward for forward speed
    action_cost = -0.5 * float(np.sum(action ** 2))  # Increased penalty for high torque usage
    smoothness_penalty = -0.1 * (np.abs(obs[6]) + np.abs(obs[7]))  # Increased penalty for joint velocity

    # Normalize components to ensure balance
    temp_action_cost = 1.0
    temp_smoothness = 1.0
    normalized_action_cost = np.exp(-np.clip(action_cost / temp_action_cost, -20.0, 20.0))
    normalized_smoothness_penalty = np.exp(-np.clip(smoothness_penalty / temp_smoothness, -20.0, 20.0))

    total = forward_velocity + normalized_action_cost + normalized_smoothness_penalty
    return {
        "total": total,
        "forward_velocity": forward_velocity,
        "action_cost": action_cost,
        "smoothness_penalty": smoothness_penalty
    }