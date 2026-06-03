def summarize(trajectory):
    import numpy as np
    
    obs = np.array([t[0] for t in trajectory])
    actions = np.array([t[1] for t in trajectory])
    next_obs = np.array([t[2] for t in trajectory])
    r_comp = [t[3] for t in trajectory]
    
    forward_velocities = next_obs[:, 3]
    avg_forward_velocity = np.mean(forward_velocities)
    std_forward_velocity = np.std(forward_velocities)
    
    energy_expended = np.sum(np.abs(actions))
    avg_energy_per_step = np.mean(np.abs(actions))
    
    episode_length = len(trajectory)
    
    summary = []
    summary.append(f"Average forward velocity: {avg_forward_velocity:.2f}")
    summary.append(f"Standard deviation of forward velocity: {std_forward_velocity:.2f}")
    summary.append(f"Total energy expended: {energy_expended:.2f}")
    summary.append(f"Average energy per step: {avg_energy_per_step:.2f}")
    summary.append(f"Episode length: {episode_length}")
    
    for k in r_comp[0].keys():
        if k == "total":
            continue
        component_values = [r[k] for r in r_comp]
        mean_value = np.mean(component_values)
        sum_value = np.sum(component_values)
        summary.append(f"Mean {k}: {mean_value:.2f}, Sum {k}: {sum_value:.2f}")
    
    return "\n".join(summary)