def reward(obs, action, next_obs) -> dict:
    theta = np.arctan2(obs[1], obs[0])  # Calculate the angle from cos and sin
    theta_dot = obs[2]                   # Angular velocity
    torque = action[0]                   # Torque applied

    # Temperature parameters for normalization
    upright_temp = 0.5
    stability_temp = 0.5
    action_cost_temp = 0.01
    swing_up_temp = 0.5

    # Reward components
    upright_reward = np.exp(-np.clip(theta ** 2 / upright_temp, -20.0, 20.0))  # Reward for being upright (theta close to 0)
    stability_penalty = -np.exp(-np.clip(theta_dot ** 2 / stability_temp, -20.0, 20.0))  # Penalty for angular velocity (stability)
    action_cost = -action_cost_temp * (torque ** 2)  # Penalty for torque usage (energy cost)

    # New component to encourage quick swinging up
    swing_up_reward = np.exp(-np.clip((1 - np.abs(obs[0])) / swing_up_temp, -20.0, 20.0))  # Reward for swinging up towards upright

    total = upright_reward + stability_penalty + action_cost + swing_up_reward

    return {
        'total': total,
        'upright_reward': upright_reward,
        'stability_penalty': stability_penalty,
        'action_cost': action_cost,
        'swing_up_reward': swing_up_reward
    }