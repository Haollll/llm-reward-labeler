def reward(obs, action, next_obs):
    # Reward for forward speed
    forward_speed = next_obs[13]  # already a velocity
    
    # Penalize large control torques (energy)
    action_cost = -0.05 * float(np.sum(np.square(action)))  # reduce effort penalty
    
    # Stability term based on height and orientation
    height = obs[0]
    stability = 0.0
    if 0.2 < height < 1.0:  # healthy height range
        stability += 1.0
    else:
        stability -= 1.0  # penalize if too low or too high
    
    # Penalize lateral drift
    lateral_drift_penalty = -0.2 * abs(obs[14])  # reduce penalty for lateral velocity
    
    # Total reward calculation
    total = forward_speed + action_cost + stability + lateral_drift_penalty
    
    return {
        "total": total,
        "forward_speed": forward_speed,
        "action_cost": action_cost,
        "stability": stability,
        "lateral_drift_penalty": lateral_drift_penalty
    }