def reward(obs, action, next_obs):
    theta = np.arctan2(obs[1], obs[0])  # Calculate the angle from cos and sin
    theta_dot = obs[2]                   # Angular velocity
    torque = action[0]                   # Torque applied

    # Reward components
    upright_reward = np.exp(-np.clip(theta ** 2, -20.0, 20.0))  # Reward for being upright (theta close to 0)
    stability_penalty = -np.exp(-np.clip(theta_dot ** 2, -20.0, 20.0))  # Penalty for angular velocity (stability)
    action_cost = -0.001 * (torque ** 2)  # Penalty for torque usage (energy cost)

    total = upright_reward + stability_penalty + action_cost

    return {
        "total": total,
        "upright_reward": upright_reward,
        "stability_penalty": stability_penalty,
        "action_cost": action_cost
    }