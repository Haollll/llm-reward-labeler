def summarize(trajectory) -> str:
    import numpy as np
    
    obs = np.array([t[0] for t in trajectory])
    actions = np.array([t[1] for t in trajectory])
    next_obs = np.array([t[2] for t in trajectory])
    
    forward_velocities = next_obs[:, 3]
    mean_forward_velocity = np.mean(forward_velocities)
    std_forward_velocity = np.std(forward_velocities)
    max_forward_velocity = np.max(forward_velocities)
    min_forward_velocity = np.min(forward_velocities)

    mean_angle_front_tip = np.mean(obs[:, 0])
    mean_angle_first_rotor = np.mean(obs[:, 1])
    mean_angle_second_rotor = np.mean(obs[:, 2])
    
    mean_torque_first_rotor = np.mean(actions[:, 0])
    mean_torque_second_rotor = np.mean(actions[:, 1])
    
    mean_angular_velocity_front_tip = np.mean(obs[:, 5])
    mean_angular_velocity_first_rotor = np.mean(obs[:, 6])
    mean_angular_velocity_second_rotor = np.mean(obs[:, 7])
    
    reward_breakdown = {k: [] for k in trajectory[0][3].keys() if k != "total"}
    
    for step in trajectory:
        for k in reward_breakdown.keys():
            reward_breakdown[k].append(step[3][k])
    
    summary_lines = []
    summary_lines.append(f"Mean forward velocity: {mean_forward_velocity:.2f}, "
                        f"Std: {std_forward_velocity:.2f}, "
                        f"Max: {max_forward_velocity:.2f}, "
                        f"Min: {min_forward_velocity:.2f}.")
    summary_lines.append(f"Mean angle of front tip: {mean_angle_front_tip:.2f}.")
    summary_lines.append(f"Mean angle of first rotor: {mean_angle_first_rotor:.2f}.")
    summary_lines.append(f"Mean angle of second rotor: {mean_angle_second_rotor:.2f}.")
    summary_lines.append(f"Mean torque of first rotor: {mean_torque_first_rotor:.2f}.")
    summary_lines.append(f"Mean torque of second rotor: {mean_torque_second_rotor:.2f}.")
    summary_lines.append(f"Mean angular velocity of front tip: {mean_angular_velocity_front_tip:.2f}.")
    summary_lines.append(f"Mean angular velocity of first rotor: {mean_angular_velocity_first_rotor:.2f}.")
    summary_lines.append(f"Mean angular velocity of second rotor: {mean_angular_velocity_second_rotor:.2f}.")

    for k in reward_breakdown:
        mean_reward = np.mean(reward_breakdown[k])
        sum_reward = np.sum(reward_breakdown[k])
        summary_lines.append(f"Mean {k} reward: {mean_reward:.2f}, Sum: {sum_reward:.2f}.")
    
    return "\n".join(summary_lines)