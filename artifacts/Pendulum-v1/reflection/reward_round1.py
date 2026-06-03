def reward(obs, action, next_obs):
    # Extract observations
    cos_theta = obs[0]  # cos(theta)
    sin_theta = obs[1]  # sin(theta)
    theta_dot = obs[2]  # angular velocity

    # Calculate the reward components
    stability = cos_theta  # Reward for being upright
    action_cost = -0.01 * np.sum(np.square(action))  # Penalty for torque usage
    angular_velocity_penalty = -0.1 * (theta_dot ** 2)  # Penalty for angular velocity

    # Total reward
    total = stability + action_cost + angular_velocity_penalty

    return {
        "total": total,
        "stability": stability,
        "action_cost": action_cost,
        "angular_velocity_penalty": angular_velocity_penalty
    }