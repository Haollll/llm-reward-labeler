def reward(obs, action, next_obs):
    forward_speed = next_obs[5]  # Reward for forward speed
    height = obs[0]               # Torso height
    torso_angle = obs[1]          # Torso angle
    action_cost = -0.05 * float(np.sum(action ** 2))  # Penalize large control torques

    # Reward for staying upright and healthy
    height_reward = 1.0 if height > 0.6 else 0.0  # Increase height threshold for reward
    angle_reward = 1.0 if abs(torso_angle) < 0.15 else 0.0  # Tighten angle condition

    # Introduce temperature scaling for forward speed
    temp_forward = 10.0
    forward_speed_reward = np.exp(-np.clip(forward_speed / temp_forward, -20.0, 20.0))

    # Total reward calculation
    total = forward_speed_reward + action_cost + height_reward + angle_reward

    return {
        "total": total,
        "forward_speed": forward_speed_reward,
        "action_cost": action_cost,
        "height_reward": height_reward,
        "angle_reward": angle_reward
    }