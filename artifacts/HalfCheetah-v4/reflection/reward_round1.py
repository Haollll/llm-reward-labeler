def reward(obs, action, next_obs):
    forward_speed = float(next_obs[8])  # reward for forward speed
    action_cost = -0.05 * float(np.sum(np.square(action)))  # reduced effort penalty
    stability = -0.3 * float(np.abs(obs[1]))  # reduced penalty for body angle deviation
    joint_velocity_penalty = -0.02 * float(np.sum(np.square(next_obs[11:17])))  # reduced penalty for joint velocities
    
    total = forward_speed + action_cost + stability + joint_velocity_penalty
    return {
        "total": total,
        "forward_speed": forward_speed,
        "action_cost": action_cost,
        "stability": stability,
        "joint_velocity_penalty": joint_velocity_penalty
    }