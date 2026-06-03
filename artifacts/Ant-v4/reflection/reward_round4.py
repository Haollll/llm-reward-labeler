def reward(obs, action, next_obs):
    # Reward for forward speed
    forward_speed = next_obs[13] * 2.0  # Increase weight further to emphasize speed
    
    # Penalize large control torques (energy)
    action_cost = -0.01 * float(np.sum(np.square(action)))  # Increase penalty to discourage high energy use
    
    # Stability term based on height and orientation
    height = obs[0]
    stability = 0.0
    if 0.35 < height < 0.75:  # Adjusted healthy height range
        stability += 0.3  # Further reduce stability reward
    else:
        stability -= 1.5  # Increase penalty if too low or too high
    
    # Penalize lateral drift
    lateral_drift_penalty = -0.3 * abs(obs[14])  # Further increase penalty for lateral velocity
    
    # Total reward calculation
    total = forward_speed + action_cost + stability + lateral_drift_penalty
    
    return {
        "total": total,
        "forward_speed": forward_speed,
        "action_cost": action_cost,
        "stability": stability,
        "lateral_drift_penalty": lateral_drift_penalty
    }