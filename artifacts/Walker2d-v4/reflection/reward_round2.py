def reward(obs, action, next_obs):
    height = obs[0]
    torso_angle = obs[1]
    forward_velocity = next_obs[8]
    
    # Reward for moving forward
    progress = forward_velocity
    
    # Survival bonus for staying upright and healthy
    height_bonus = 1.0 if 0.7 < height < 2.0 else -1.0
    angle_bonus = 1.0 if -0.2 < torso_angle < 0.2 else -1.0
    survival_bonus = height_bonus + angle_bonus
    
    # Penalty for large control torques (energy)
    action_cost_temp = 5.0  # Temperature for action cost
    action_cost = -np.clip(np.exp(-np.sum(action ** 2) / action_cost_temp), 0, 1)
    
    # Total reward calculation
    total = progress + survival_bonus + action_cost
    
    return {
        "total": total,
        "progress": progress,
        "survival_bonus": survival_bonus,
        "action_cost": action_cost
    }