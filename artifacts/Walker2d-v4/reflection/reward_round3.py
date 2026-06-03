def reward(obs, action, next_obs):
    forward_speed = next_obs[8]  # Reward for forward speed
    action_cost = -0.01 * float(np.sum(np.square(action)))  # Reduce effort penalty further
    
    # Stability and survival terms
    height = obs[0]
    torso_angle = obs[1]
    
    # Reward for staying upright and healthy
    stability = 0.0
    if height > 0.6:  # Increase healthy height threshold
        stability += 1.0
    stability += np.exp(-np.clip(np.abs(torso_angle), 0, 0.5))  # Penalize large angles more
    
    # Increase survival bonus to encourage longer episodes
    survival_bonus = 0.5
    
    total = forward_speed + action_cost + stability + survival_bonus
    return {
        "total": total,
        "forward_speed": forward_speed,
        "action_cost": action_cost,
        "stability": stability,
        "survival_bonus": survival_bonus
    }