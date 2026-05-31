def summarize(trajectory):
    import numpy as np

    obs = np.array([t[0] for t in trajectory])
    actions = np.array([t[1] for t in trajectory])
    next_obs = np.array([t[2] for t in trajectory])
    
    # Extract relevant observations
    torso_heights = obs[:, 0]
    forward_velocities = next_obs[:, 13]
    lateral_velocities = next_obs[:, 14]
    angular_velocities = obs[:, 16:19]
    joint_angles = obs[:, 5:13]
    joint_velocities = obs[:, 19:27]
    
    # Calculate behavioral features
    mean_height = np.mean(torso_heights)
    std_height = np.std(torso_heights)
    mean_forward_velocity = np.mean(forward_velocities)
    max_forward_velocity = np.max(forward_velocities)
    mean_lateral_velocity = np.mean(np.abs(lateral_velocities))
    mean_joint_angles = np.mean(joint_angles, axis=0)
    mean_joint_velocities = np.mean(joint_velocities, axis=0)
    
    summary = (
        f"Average torso height: {mean_height:.2f} (std: {std_height:.2f}). "
        f"Average forward velocity: {mean_forward_velocity:.2f} (max: {max_forward_velocity:.2f}). "
        f"Average lateral velocity: {mean_lateral_velocity:.2f}. "
        f"Mean joint angles: {mean_joint_angles}. "
        f"Mean joint velocities: {mean_joint_velocities}."
    )
    
    # Reward breakdown
    r_comp = trajectory[0][3]  # Assuming r_comp is the same for all steps
    for k in r_comp:
        if k == "total":
            continue
        per_step_mean = np.mean([t[3][k] for t in trajectory])
        trajectory_sum = np.sum([t[3][k] for t in trajectory])
        summary += f" Mean {k}: {per_step_mean:.2f}, Total {k}: {trajectory_sum:.2f}."
    
    return summary.strip()