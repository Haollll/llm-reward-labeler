def summarize(trajectory):
    import numpy as np
    
    episode_length = len(trajectory)
    if episode_length == 0:
        return "No data available for summary."
    
    heights = np.array([obs[0] for obs, action, next_obs, r_comp, done in trajectory])
    forward_velocities = np.array([next_obs[13] for obs, action, next_obs, r_comp, done in trajectory])
    lateral_velocities = np.array([next_obs[14] for obs, action, next_obs, r_comp, done in trajectory])
    actions = np.array([action for obs, action, next_obs, r_comp, done in trajectory])
    
    height_mean = np.mean(heights)
    height_std = np.std(heights)
    forward_velocity_mean = np.mean(forward_velocities)
    forward_velocity_std = np.std(forward_velocities)
    lateral_velocity_mean = np.mean(np.abs(lateral_velocities))
    lateral_velocity_std = np.std(np.abs(lateral_velocities))
    energy_expended = np.sum(np.abs(actions))
    
    healthy_steps = np.sum((heights > 0.35) & (heights < 0.75))
    health_ratio = healthy_steps / episode_length
    
    summary = (
        f"Episode length: {episode_length}\n"
        f"Average height: {height_mean:.2f} (std: {height_std:.2f})\n"
        f"Average forward velocity: {forward_velocity_mean:.2f} (std: {forward_velocity_std:.2f})\n"
        f"Average lateral velocity: {lateral_velocity_mean:.2f} (std: {lateral_velocity_std:.2f})\n"
        f"Total energy expended: {energy_expended:.2f}\n"
        f"Healthy steps ratio: {health_ratio:.2f}"
    )
    
    r_comp = trajectory[0][3]
    for k in r_comp:
        if k == "total":
            continue
        per_step_mean = np.mean([r_comp[k] for obs, action, next_obs, r_comp, done in trajectory])
        trajectory_sum = np.sum([r_comp[k] for obs, action, next_obs, r_comp, done in trajectory])
        summary += f"\nMean {k}: {per_step_mean:.2f}, Sum {k}: {trajectory_sum:.2f}"
    
    return summary