def reward(obs, action, next_obs):
    forward_velocity = next_obs[3]  # Reward for forward speed
    action_cost = -0.02 * float(np.sum(np.square(action)))  # Further reduce penalty for control torques
    stability = -0.2 * (np.abs(obs[1]) + np.abs(obs[2]))  # Increase penalty for excessive joint angles

    total = forward_velocity + action_cost + stability
    return {
        "total": total,
        "forward_velocity": forward_velocity,
        "action_cost": action_cost,
        "stability": stability
    }