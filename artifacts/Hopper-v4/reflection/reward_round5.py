def reward(obs, action, next_obs):
    # Reward for forward speed
    forward_speed = next_obs[5]  # x-axis velocity of the torso
    forward_speed_reward = 10.0 * forward_speed  # Increase weight to encourage forward motion

    # Penalize large control torques (energy cost)
    action_cost = -0.2 * float(np.sum(np.square(action)))  # Increase penalty to discourage excessive energy use

    # Stability and survival rewards
    height = obs[0]  # z-position of the torso
    torso_angle = obs[1]  # angle of the torso
    stability = 0.0

    # Reward for staying upright (torso angle close to 0)
    if -0.5 <= torso_angle <= 0.5:
        stability += 1.0  # Keep reward for staying upright

    # Penalize for being too low (height)
    if height < 0.5:
        stability -= 1.0  # Keep penalty for being too low

    # Total reward calculation
    total = forward_speed_reward + action_cost + stability
    return {
        "total": total,
        "forward_speed_reward": forward_speed_reward,
        "action_cost": action_cost,
        "stability": stability
    }