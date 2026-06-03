def reward(obs, action, next_obs):
    forward_speed = 0.1 * float(next_obs[8])  # Increase scale to make it more influential
    action_cost = -0.05 * float(np.sum(np.square(action)))  # Reduce penalty to allow more exploration
    stability = -0.5 * float(np.abs(obs[1]))  # Keep stability penalty as is
    joint_velocity_penalty = -0.02 * float(np.sum(np.square(next_obs[11:17])))  # Reduce penalty to encourage more movement
    
    total = forward_speed + action_cost + stability + joint_velocity_penalty
    return {
        "total": total,
        "forward_speed": forward_speed,
        "action_cost": action_cost,
        "stability": stability,
        "joint_velocity_penalty": joint_velocity_penalty
    }