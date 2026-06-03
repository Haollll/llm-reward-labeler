def reward(obs, action, next_obs):
    forward_speed = 0.01 * float(next_obs[8])  # scale down forward speed reward
    action_cost = -0.1 * float(np.sum(np.square(action)))  # keep action cost as is
    stability = -0.5 * float(np.abs(obs[1]))  # keep stability penalty as is
    joint_velocity_penalty = -0.05 * float(np.sum(np.square(next_obs[11:17])))  # keep joint velocity penalty as is
    
    total = forward_speed + action_cost + stability + joint_velocity_penalty
    return {
        "total": total,
        "forward_speed": forward_speed,
        "action_cost": action_cost,
        "stability": stability,
        "joint_velocity_penalty": joint_velocity_penalty
    }