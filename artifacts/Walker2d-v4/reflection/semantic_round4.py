def summarize(trajectory) -> str:
    import numpy as np
    
    obs = np.array([t[0] for t in trajectory])
    next_obs = np.array([t[2] for t in trajectory])
    actions = np.array([t[1] for t in trajectory])
    
    # Extract relevant observations
    torso_height = obs[:, 0]
    torso_angle = obs[:, 1]
    forward_velocity = next_obs[:, 8]
    
    # Calculate behavioral features
    mean_height = np.mean(torso_height)
    std_height = np.std(torso_height)
    mean_angle = np.mean(torso_angle)
    std_angle = np.std(torso_angle)
    mean_forward_velocity = np.mean(forward_velocity)
    max_forward_velocity = np.max(forward_velocity)
    
    # Calculate action statistics
    mean_action = np.mean(actions, axis=0)
    std_action = np.std(actions, axis=0)
    
    # Prepare the summary
    summary = []
    summary.append(f"Average torso height: {mean_height:.2f} (std: {std_height:.2f})")
    summary.append(f"Average torso angle: {mean_angle:.2f} (std: {std_angle:.2f})")
    summary.append(f"Average forward velocity: {mean_forward_velocity:.2f}, max: {max_forward_velocity:.2f}")
    summary.append(f"Average action torque: {mean_action}, action torque std: {std_action}")
    
    # Reward breakdown
    r_comp = trajectory[0][3]  # Assuming r_comp is the same for all steps
    for k in r_comp:
        if k == "total":
            continue
        mean_reward = np.mean([t[3][k] for t in trajectory])
        sum_reward = sum(t[3][k] for t in trajectory)
        summary.append(f"Mean {k} reward: {mean_reward:.2f}, total {k} reward: {sum_reward:.2f}")
    
    return "\n".join(summary)