def summarize(trajectory):
    import numpy as np
    
    obs = np.array([t[0] for t in trajectory])
    actions = np.array([t[1] for t in trajectory])
    next_obs = np.array([t[2] for t in trajectory])
    
    cos_theta = obs[:, 0]
    sin_theta = obs[:, 1]
    angular_velocity = obs[:, 2]
    
    mean_cos_theta = np.mean(cos_theta)
    std_cos_theta = np.std(cos_theta)
    mean_sin_theta = np.mean(sin_theta)
    std_sin_theta = np.std(sin_theta)
    mean_angular_velocity = np.mean(angular_velocity)
    std_angular_velocity = np.std(angular_velocity)
    
    mean_action = np.mean(actions)
    std_action = np.std(actions)
    
    summary = []
    summary.append(f"Mean cos(theta): {mean_cos_theta:.4f}, Std cos(theta): {std_cos_theta:.4f}")
    summary.append(f"Mean sin(theta): {mean_sin_theta:.4f}, Std sin(theta): {std_sin_theta:.4f}")
    summary.append(f"Mean angular velocity: {mean_angular_velocity:.4f}, Std angular velocity: {std_angular_velocity:.4f}")
    summary.append(f"Mean torque applied: {mean_action:.4f}, Std torque: {std_action:.4f}")
    
    for k in trajectory[0][3]:  # r_comp is in the 4th element of the tuple
        if k == "total":
            continue
        component_values = np.array([t[3][k] for t in trajectory])
        mean_component = np.mean(component_values)
        sum_component = np.sum(component_values)
        summary.append(f"Mean {k}: {mean_component:.4f}, Sum {k}: {sum_component:.4f}")
    
    return "\n".join(summary)