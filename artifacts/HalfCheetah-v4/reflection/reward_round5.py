def reward(obs, action, next_obs):
    forward_speed = 0.05 * float(next_obs[8])  # Reduce scale to balance with other components
    action_cost = -0.1 * float(np.sum(np.square(action)))  # Increase penalty to influence policy
    stability = -0.5 * float(np.abs(obs[1]))  # Keep stability penalty as is
    joint_velocity_penalty = -0.05 * float(np.sum(np.square(next_obs[11:17])))  # Increase penalty to influence policy
    
    total = forward_speed + action_cost + stability + joint_velocity_penalty
    return {
        "total": total,
        "forward_speed": forward_speed,
        "action_cost": action_cost,
        "stability": stability,
        "joint_velocity_penalty": joint_velocity_penalty
    }