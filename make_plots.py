"""Generate the PDF plot suite from saved training artifacts.

Per-env plots (episodic return per round, alpha vs return, reward-model loss,
reward components, reflection changes, label-buffer growth):

    python make_plots.py --env HalfCheetah-v4
    python make_plots.py --env HalfCheetah-v4 Ant-v4        # several envs

Cross-env comparison bars (return / length / success, pipeline vs RL-Zoo3):

    python make_plots.py --cross-env Pendulum-v1 Swimmer-v4 HalfCheetah-v4 \
                                     Hopper-v4 Walker2d-v4 Ant-v4
"""

import argparse

import helper  # noqa: F401  (env-warning suppression)
import plots
from helper import SUPPORTED_ENVS

DEFAULT_ENVS = SUPPORTED_ENVS


def main() -> None:
    p = argparse.ArgumentParser(description="Generate the PDF plot suite")
    p.add_argument("--env", nargs="*", default=None,
                   help="Envs to make per-env plots for")
    p.add_argument("--cross-env", nargs="*", default=None,
                   help="Envs to include in the cross-env comparison bars")
    p.add_argument("--artifact-dir", default="artifacts")
    args = p.parse_args()

    if not args.env and args.cross_env is None:
        args.env = DEFAULT_ENVS  # default: per-env plots for all v4 envs found

    if args.env:
        for env_id in args.env:
            print(f"[per-env] {env_id}")
            for path in plots.make_all_per_env(env_id, args.artifact_dir):
                print(f"  → {path}")

    if args.cross_env is not None:
        envs = args.cross_env or DEFAULT_ENVS
        print(f"[cross-env] {', '.join(envs)}")
        try:
            for path in plots.plot_cross_env_bars(envs, args.artifact_dir):
                print(f"  → {path}")
        except Exception as e:
            print(f"  [skip] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
