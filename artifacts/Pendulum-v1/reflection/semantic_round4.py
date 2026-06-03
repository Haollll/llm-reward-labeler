def summarize(trajectory):
    import numpy as np
    
    obs = np.array([t[0] for t in trajectory])
    actions = np.array([t[1] for t in trajectory])
    rewards = np.array([t[3] for t in trajectory])
    
    # Calculate behavioral features
    cos_theta = obs[:, 0]
    sin_theta = obs[:, 1]
    angular_velocity = obs[:, 2]
    
    mean_cos_theta = np.mean(cos_theta)
    mean_sin_theta = np.mean(sin_theta)
    mean_angular_velocity = np.mean(angular_velocity)
    max_cos_theta = np.max(cos_theta)
    min_cos_theta = np.min(cos_theta)
    torque_used = np.sum(np.square(actions))
    
    episode_length = len(trajectory)
    
    summary = (
        f"Mean cos(theta): {mean_cos_theta:.4f}\n"
        f"Mean sin(theta): {mean_sin_theta:.4f}\n"
        f"Mean angular velocity: {mean_angular_velocity:.4f}\n"
        f"Max cos(theta): {max_cos_theta:.4f}\n"
        f"Min cos(theta): {min_cos_theta:.4f}\n"
        f"Torque used (energy expended): {torque_used:.4f}\n"
        f"Episode length: {episode_length}\n"
    )
    
    # Reward breakdown
    r_comp = trajectory[0][3]  # Assuming r_comp is the same for all steps
    for k in r_comp:
        if k == "total":
            continue
        mean_reward = np.mean([t[3][k] for t in trajectory])
        sum_reward = np.sum([t[3][k] for t in trajectory])
        summary += f"Mean {k}: {mean_reward:.4f}, Sum {k}: {sum_reward:.4f}\n"
    
    return summary.strip()