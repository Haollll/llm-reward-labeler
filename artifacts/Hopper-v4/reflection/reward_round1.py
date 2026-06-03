def reward(obs, action, next_obs):
    # Reward for forward speed
    forward_speed = next_obs[5]  # x-axis velocity of the torso
    
    # Penalize large control torques (energy cost)
    action_cost = -0.05 * float(np.sum(np.square(action)))  # Reduced penalty

    # Stability and survival rewards
    height = obs[0]  # z-position of the torso
    torso_angle = obs[1]  # angle of the torso
    stability = 0.0

    # Reward for staying upright (torso angle close to 0)
    if -0.5 <= torso_angle <= 0.5:
        stability += 0.5  # Reduced reward

    # Penalize for being too low (height)
    if height < 0.5:
        stability -= 0.5  # Reduced penalty

    # Total reward calculation
    total = forward_speed + action_cost + stability
    return {
        "total": total,
        "forward_speed": forward_speed,
        "action_cost": action_cost,
        "stability": stability
    }