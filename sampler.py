from typing import Callable, List, Optional, Tuple

import numpy as np

from env_setup import collect_trajectory
from llm_utils import compare_trajectories
from reward_model import RewardModel, traj_to_segment

Trajectory = List[Tuple]


# ─────────────────────────────────────────────────────────────
# Sampler
# ─────────────────────────────────────────────────────────────

class Sampler:
    """
    Collects trajectory pairs and obtains LLM preference labels.

    Two query strategies:
      uniform — random pairs (used during cold start)
      active  — disagreement sampling: collect CANDIDATE_RATIO × n
                candidates, query only the most informative pairs
    """

    CANDIDATE_RATIO = 5

    def __init__(
        self,
        env,
        policy_fn: Callable,
        reward_model: RewardModel,
        semantic_fn: Callable,
        task: str,
        segment_length: int = 50,
        llm_model: str = "gpt-4o-mini",
        verbose: bool = True,
        reward_fn: Optional[Callable] = None,
    ):
        self.env            = env
        self.policy_fn      = policy_fn
        self.reward_model   = reward_model
        self.semantic_fn    = semantic_fn
        self.task           = task
        self.seg_len        = segment_length
        self.llm_model      = llm_model
        self.verbose        = verbose
        self.reward_fn      = reward_fn

    # ── public ───────────────────────────────────────────────

    def collect_and_label(self, n_queries: int, use_active: bool = True) -> int:
        """
        Collect trajectory pairs, obtain labels, add to reward model buffer.
        Returns the number of pairs successfully added.
        """
        trajs_a, trajs_b = self._collect_candidates(n_queries, use_active)
        segs_a = np.array([traj_to_segment(t, self.seg_len) for t in trajs_a])
        segs_b = np.array([traj_to_segment(t, self.seg_len) for t in trajs_b])

        if use_active and len(self.reward_model.buffer) >= 10:
            segs_a, segs_b, trajs_a, trajs_b = self._select_by_disagreement(
                segs_a, segs_b, trajs_a, trajs_b, n=n_queries
            )
            if self.verbose:
                print(f"  Active learning: selected top-{n_queries} disagreed pairs")

        added = 0
        for i in range(len(trajs_a)):
            label = self._query_label(trajs_a[i], trajs_b[i])
            self.reward_model.add(segs_a[i], segs_b[i], float(label))
            added += 1

        return added

    # ── private ──────────────────────────────────────────────

    def _collect_candidates(
        self, n_queries: int, use_active: bool
    ) -> Tuple[List[Trajectory], List[Trajectory]]:
        n = n_queries * self.CANDIDATE_RATIO if use_active else n_queries
        trajs_a = [collect_trajectory(self.env, self.policy_fn, self.seg_len, reward_fn=self.reward_fn) for _ in range(n)]
        trajs_b = [collect_trajectory(self.env, self.policy_fn, self.seg_len, reward_fn=self.reward_fn) for _ in range(n)]
        return trajs_a, trajs_b

    def _select_by_disagreement(
        self,
        segs_a: np.ndarray,
        segs_b: np.ndarray,
        trajs_a: List[Trajectory],
        trajs_b: List[Trajectory],
        n: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[Trajectory], List[Trajectory]]:
        scores = self.reward_model.disagreement(segs_a, segs_b)
        top_k  = (-scores).argsort()[:n]
        return (
            segs_a[top_k],
            segs_b[top_k],
            [trajs_a[i] for i in top_k],
            [trajs_b[i] for i in top_k],
        )

    def _query_label(self, traj_a: Trajectory, traj_b: Trajectory) -> int:
        label, explanation = compare_trajectories(
            traj_a, traj_b,
            semantic_fn=self.semantic_fn,
            task_description=self.task,
            model=self.llm_model,
        )
        if self.verbose:
            winner = "A" if label == 1 else "B"
            print(f"    LLM → {winner} | {explanation[:70]}")
        return label