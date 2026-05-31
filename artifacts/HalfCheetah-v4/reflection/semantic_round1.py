def summarize(trajectory) -> str:
    import numpy as np

    obs = np.array([t[0] for t in trajectory])
    actions = np.array([t[1] for t in trajectory])
    next_obs = np.array([t[2] for t in trajectory])
    r_comp = [t[3] for t in trajectory]

    mean_z_position = np.mean(obs[:, 0])
    std_z_position = np.std(obs[:, 0])
    mean_body_angle = np.mean(obs[:, 1])
    std_body_angle = np.std(obs[:, 1])
    mean_forward_velocity = np.mean(next_obs[:, 8])
    std_forward_velocity = np.std(next_obs[:, 8])
    mean_vertical_velocity = np.mean(next_obs[:, 9])
    std_vertical_velocity = np.std(next_obs[:, 9])
    mean_joint_angles = np.mean(obs[:, 2:8], axis=0)
    mean_joint_velocities = np.mean(obs[:, 10:17], axis=0)

    reward_breakdown = {k: (np.mean([r[k] for r in r_comp]), np.sum([r[k] for r in r_comp])) for k in r_comp[0] if k != 'total'}

    summary = (
        f'Mean z-position: {mean_z_position:.2f} (std: {std_z_position:.2f}). '
        f'Mean body angle: {mean_body_angle:.2f} (std: {std_body_angle:.2f}). '
        f'Mean forward velocity: {mean_forward_velocity:.2f} (std: {std_forward_velocity:.2f}). '
        f'Mean vertical velocity: {mean_vertical_velocity:.2f} (std: {std_vertical_velocity:.2f}). '
        f'Mean joint angles: {mean_joint_angles}. '
        f'Mean joint angular velocities: {mean_joint_velocities}.'
    )

    for k in reward_breakdown:
        mean_value, total_value = reward_breakdown[k]
        summary += f' {k} mean: {mean_value:.2f}, sum: {total_value:.2f}.'

    return summary.strip()