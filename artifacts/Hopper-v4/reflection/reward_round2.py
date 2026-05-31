def reward(obs, action, next_obs) -> dict:
    forward_speed = next_obs[5]  # Reward for forward speed
    height = obs[0]               # Torso height
    torso_angle = obs[1]          # Torso angle
    action_cost = -0.1 * float(np.sum(action ** 2))  # Penalty for large control torques

    # Normalize rewards with temperature
    temp_forward = 10.0
    temp_height = 5.0
    temp_angle = 5.0
    temp_action = 20.0

    # Reward for staying upright and healthy
    height_reward = 1.0 if height > 0.5 else 0.0  # Reward for being above a certain height
    angle_reward = 1.0 if abs(torso_angle) < 0.2 else 0.0  # Reward for keeping torso angle small

    # Total reward calculation with normalization
    total = (np.exp(forward_speed / temp_forward) +
             np.exp(height_reward / temp_height) +
             np.exp(angle_reward / temp_angle) +
             action_cost)

    return {
        "total": total,
        "forward_speed": forward_speed,
        "action_cost": action_cost,
        "height_reward": height_reward,
        "angle_reward": angle_reward
    }