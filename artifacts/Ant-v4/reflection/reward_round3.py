def reward(obs, action, next_obs):
    # Reward for forward speed
    forward_speed = next_obs[13] * 1.5  # Increase weight to emphasize speed
    
    # Penalize large control torques (energy)
    action_cost = -0.005 * float(np.sum(np.square(action)))  # Reduce penalty further
    
    # Stability term based on height and orientation
    height = obs[0]
    stability = 0.0
    if 0.35 < height < 0.75:  # Adjusted healthy height range
        stability += 0.5  # Reduce stability reward
    else:
        stability -= 1.0  # Reduce penalty if too low or too high
    
    # Penalize lateral drift
    lateral_drift_penalty = -0.2 * abs(obs[14])  # Increase penalty for lateral velocity
    
    # Total reward calculation
    total = forward_speed + action_cost + stability + lateral_drift_penalty
    
    return {
        "total": total,
        "forward_speed": forward_speed,
        "action_cost": action_cost,
        "stability": stability,
        "lateral_drift_penalty": lateral_drift_penalty
    }