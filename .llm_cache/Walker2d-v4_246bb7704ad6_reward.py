def reward(obs, action, next_obs):
    forward_speed = next_obs[8]  # Reward for forward speed
    action_cost = -0.1 * float(np.sum(np.square(action)))  # Effort penalty
    
    # Stability and survival terms
    height = obs[0]
    torso_angle = obs[1]
    
    # Reward for staying upright (torso angle close to 0) and healthy (height above a threshold)
    stability = 0.0
    if height > 0.5:  # Healthy height threshold
        stability += 1.0
    stability += np.exp(-np.clip(np.abs(torso_angle), 0, 1))  # Penalize large angles
    
    total = forward_speed + action_cost + stability
    return {
        "total": total,
        "forward_speed": forward_speed,
        "action_cost": action_cost,
        "stability": stability
    }