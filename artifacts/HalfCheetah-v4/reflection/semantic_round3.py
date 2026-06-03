def summarize(trajectory):
    import numpy as np
    
    obs = np.array([t[0] for t in trajectory])
    next_obs = np.array([t[2] for t in trajectory])
    actions = np.array([t[1] for t in trajectory])
    r_comp = [t[3] for t in trajectory]
    
    episode_length = len(trajectory)
    
    avg_forward_velocity = np.mean(next_obs[:, 8]) if episode_length > 0 else 0
    height_stability = np.std(obs[:, 0]) if episode_length > 0 else 0
    avg_joint_angles = np.mean(obs[:, 2:8], axis=0) if episode_length > 0 else np.zeros(6)
    avg_energy_expended = np.mean(np.abs(actions)) if episode_length > 0 else 0
    healthy_steps = np.sum(np.all(np.abs(obs[:, 2:8]) < 1.5, axis=1))  # Assuming healthy if joint angles are within [-1.5, 1.5]
    
    summary = []
    summary.append(f"Average forward velocity: {avg_forward_velocity:.2f}")
    summary.append(f"Height stability (std): {height_stability:.2f}")
    summary.append(f"Average joint angles: {', '.join(f'{angle:.2f}' for angle in avg_joint_angles)}")
    summary.append(f"Average energy expended: {avg_energy_expended:.2f}")
    summary.append(f"Healthy steps: {healthy_steps}/{episode_length}")
    
    r_comp_means = {k: np.mean([r[k] for r in r_comp]) for k in r_comp[0] if k != "total"}
    r_comp_sums = {k: np.sum([r[k] for r in r_comp]) for k in r_comp[0] if k != "total"}
    
    summary.append("Reward breakdown:")
    for k in r_comp_means:
        summary.append(f"  {k} mean: {r_comp_means[k]:.2f}, sum: {r_comp_sums[k]:.2f}")
    
    return "\n".join(summary)