def reward(obs, action, next_obs):
    # Extract observations
    cos_theta = obs[0]  # cos(theta)
    sin_theta = obs[1]  # sin(theta)
    theta_dot = obs[2]  # angular velocity

    # Calculate the reward components
    stability = 5 * (cos_theta - 1)  # Reduce weight for being upright
    action_cost = -0.1 * np.sum(np.square(action))  # Keep penalty for torque usage
    angular_velocity_penalty = -0.5 * (theta_dot ** 2)  # Keep penalty for angular velocity

    # Total reward
    total = stability + action_cost + angular_velocity_penalty

    return {
        "total": total,
        "stability": stability,
        "action_cost": action_cost,
        "angular_velocity_penalty": angular_velocity_penalty
    }