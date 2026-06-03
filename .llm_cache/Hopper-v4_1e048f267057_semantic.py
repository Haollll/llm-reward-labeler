def summarize(trajectory):
    import numpy as np
    
    obs = np.array([t[0] for t in trajectory])
    actions = np.array([t[1] for t in trajectory])
    next_obs = np.array([t[2] for t in trajectory])
    r_comp = [t[3] for t in trajectory]
    done = [t[4] for t in trajectory]
    
    episode_length = len(trajectory)
    
    # Behavioural features
    avg_forward_velocity = np.mean(next_obs[:, 5]) if episode_length > 0 else 0
    height_stability = np.std(obs[:, 0]) if episode_length > 0 else 0
    avg_height = np.mean(obs[:, 0]) if episode_length > 0 else 0
    avg_torso_angle = np.mean(obs[:, 1]) if episode_length > 0 else 0
    avg_energy_expenditure = np.mean(np.abs(actions)) if episode_length > 0 else 0
    healthy_steps = np.sum((obs[:, 0] > 0.2) & (np.abs(obs[:, 1]) < 0.5))  # Assuming healthy range
    total_healthy_steps = np.sum((obs[:, 0] > 0.2) & (np.abs(obs[:, 1]) < 0.5))
    
    summary = (
        f"Average forward velocity: {avg_forward_velocity:.2f}\n"
        f"Height stability (std): {height_stability:.2f}\n"
        f"Average height: {avg_height:.2f}\n"
        f"Average torso angle: {avg_torso_angle:.2f}\n"
        f"Average energy expenditure: {avg_energy_expenditure:.2f}\n"
        f"Healthy steps: {healthy_steps}/{episode_length}\n"
    )
    
    # Per-component reward breakdown
    r_comp_means = {k: np.mean([r[k] for r in r_comp]) for k in r_comp[0] if k != "total"}
    r_comp_sums = {k: np.sum([r[k] for r in r_comp]) for k in r_comp[0] if k != "total"}
    
    for k in r_comp_means:
        summary += f"Mean {k}: {r_comp_means[k]:.2f}, Sum {k}: {r_comp_sums[k]:.2f}\n"
    
    return summary.strip()