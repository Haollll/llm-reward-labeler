"""Generate the v2 plot suite from saved artifacts.

    python make_plots.py --env HalfCheetah-v4
    python make_plots.py --env Pendulum-v1 Swimmer-v4 HalfCheetah-v4 Hopper-v4 Walker2d-v4 Ant-v4
"""

import argparse

import plots
from helper import SUPPORTED_ENVS


def main() -> None:
    p = argparse.ArgumentParser(description="Generate v2 plots")
    p.add_argument("--env", nargs="+", default=SUPPORTED_ENVS,
                   help="One or more env ids to plot")
    p.add_argument("--artifact-dir", default="artifacts")
    args = p.parse_args()

    for env_id in args.env:
        print(f"=== {env_id} ===")
        paths = plots.make_all_per_env(env_id, args.artifact_dir)
        for pth in paths:
            print(f"  saved {pth}")


if __name__ == "__main__":
    main()
