def reward(obs, action, next_obs):
    # Extract observations
    cos_theta = obs[0]  # cos(theta)
    sin_theta = obs[1]  # sin(theta)
    theta_dot = obs[2]  # angular velocity

    # Calculate the reward components
    stability = 10 * (cos_theta - 1)  # Reward for being upright, rescaled
    action_cost = -0.1 * np.sum(np.square(action))  # Increased penalty for torque usage
    angular_velocity_penalty = -0.5 * (theta_dot ** 2)  # Increased penalty for angular velocity

    # Total reward
    total = stability + action_cost + angular_velocity_penalty

    return {
        "total": total,
        "stability": stability,
        "action_cost": action_cost,
        "angular_velocity_penalty": angular_velocity_penalty
    }