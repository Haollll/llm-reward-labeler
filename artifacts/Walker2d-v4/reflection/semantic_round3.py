def summarize(trajectory):
    episode_length = len(trajectory)
    
    if episode_length == 0:
        return "No data available."
    
    heights = np.array([obs[0] for obs, action, next_obs, r_comp, done in trajectory])
    torso_angles = np.array([obs[1] for obs, action, next_obs, r_comp, done in trajectory])
    forward_velocities = np.array([next_obs[8] for obs, action, next_obs, r_comp, done in trajectory])
    actions = np.array([action for obs, action, next_obs, r_comp, done in trajectory])
    
    height_mean = np.mean(heights)
    height_std = np.std(heights)
    angle_mean = np.mean(torso_angles)
    angle_std = np.std(torso_angles)
    forward_velocity_mean = np.mean(forward_velocities)
    forward_velocity_std = np.std(forward_velocities)
    action_mean = np.mean(np.abs(actions), axis=0)
    action_std = np.std(actions, axis=0)
    
    healthy_steps = np.sum((heights > 0.6) & (np.abs(torso_angles) < 0.5))
    healthy_ratio = healthy_steps / episode_length
    
    summary = (
        f"Average height: {height_mean:.2f} ± {height_std:.2f}\n"
        f"Average torso angle: {angle_mean:.2f} ± {angle_std:.2f}\n"
        f"Average forward velocity: {forward_velocity_mean:.2f} ± {forward_velocity_std:.2f}\n"
        f"Healthy steps ratio: {healthy_ratio:.2%}\n"
        f"Average action torque: {action_mean.mean():.2f} ± {action_std.mean():.2f}\n"
    )
    
    r_comp_summary = []
    for k in trajectory[0][3]:
        if k == "total":
            continue
        r_values = np.array([r_comp[k] for _, _, _, r_comp, _ in trajectory])
        r_mean = np.mean(r_values)
        r_sum = np.sum(r_values)
        r_comp_summary.append(f"Mean {k}: {r_mean:.2f}, Sum {k}: {r_sum:.2f}")
    
    summary += "\n".join(r_comp_summary)
    
    return summary.strip()