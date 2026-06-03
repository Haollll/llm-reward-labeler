def reward(obs, action, next_obs):
    # Reward for forward speed
    forward_speed = next_obs[13] * 1.5  # Slightly reduce the weight to balance with other components
    
    # Penalize large control torques (energy)
    action_cost = -0.05 * float(np.sum(np.square(action)))  # Increase penalty to discourage high energy use
    
    # Stability term based on height and orientation
    height = obs[0]
    stability = 0.0
    if 0.35 < height < 0.75:
        stability += 0.5  # Increase stability reward to encourage staying upright
    else:
        stability -= 2.0  # Increase penalty if too low or too high
    
    # Penalize lateral drift
    lateral_drift_penalty = -0.5 * abs(obs[14])  # Increase penalty for lateral velocity
    
    # Survival bonus
    survival_bonus = 1.0  # Encourage staying alive
    
    # Total reward calculation
    total = forward_speed + action_cost + stability + lateral_drift_penalty + survival_bonus
    
    return {
        "total": total,
        "forward_speed": forward_speed,
        "action_cost": action_cost,
        "stability": stability,
        "lateral_drift_penalty": lateral_drift_penalty,
        "survival_bonus": survival_bonus
    }