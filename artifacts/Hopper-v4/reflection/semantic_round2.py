def summarize(trajectory) -> str:
    import numpy as np
    
    obs = np.array([t[0] for t in trajectory])
    actions = np.array([t[1] for t in trajectory])
    next_obs = np.array([t[2] for t in trajectory])
    
    torso_heights = obs[:, 0]
    torso_angles = obs[:, 1]
    forward_velocities = next_obs[:, 5]
    
    mean_height = np.mean(torso_heights)
    std_height = np.std(torso_heights)
    min_height = np.min(torso_heights)
    max_height = np.max(torso_heights)
    
    mean_angle = np.mean(torso_angles)
    std_angle = np.std(torso_angles)
    min_angle = np.min(torso_angles)
    max_angle = np.max(torso_angles)
    
    mean_forward_velocity = np.mean(forward_velocities)
    std_forward_velocity = np.std(forward_velocities)
    
    mean_action = np.mean(actions, axis=0)
    std_action = np.std(actions, axis=0)
    
    r_comp = {k: t[3] for t in trajectory for k in t[3].keys()}
    reward_breakdown = []
    for k in r_comp:
        if k == "total":
            continue
        per_step_mean = np.mean([t[3][k] for t in trajectory])
        trajectory_sum = np.sum([t[3][k] for t in trajectory])
        reward_breakdown.append(f"{k} mean: {per_step_mean:.2f}, sum: {trajectory_sum:.2f}")
    
    summary = (
        f"Mean torso height: {mean_height:.2f} (std: {std_height:.2f}, min: {min_height:.2f}, max: {max_height:.2f}).\n"
        f"Mean torso angle: {mean_angle:.2f} (std: {std_angle:.2f}, min: {min_angle:.2f}, max: {max_angle:.2f}).\n"
        f"Mean forward velocity: {mean_forward_velocity:.2f} (std: {std_forward_velocity:.2f}).\n"
        f"Mean action torques: {mean_action}, action std: {std_action}.\n"
        + "\n".join(reward_breakdown)
    )
    
    return summary