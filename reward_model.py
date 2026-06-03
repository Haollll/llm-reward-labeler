"""Bradley-Terry reward model for v2.

Design choices (intentionally simpler than v1):
  * SINGLE network (no ensemble).
  * Trains on FULL trajectories (no fixed-length segments). The score of a
    trajectory is the sum of per-step rewards over its real length; pairs are
    batched with right-padding + a length mask so padding contributes nothing.
  * Uniform random pairs (no active / disagreement sampling).

The Bradley-Terry loss for a labelled pair (A, B, y) is the cross-entropy of
softmax([R_A, R_B]) against the preferred index, where R = sum_t f(s_t, a_t).

`train()` returns the per-epoch loss list so the trainer can plot loss at every
epoch of every round across both phases.
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from llm import describe_env

DEVICE = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)


def _space_dim(space_desc: dict) -> int:
    if "shape" in space_desc:
        dim = 1
        for s in space_desc["shape"]:
            dim *= int(s)
        return dim
    if "n" in space_desc:
        return int(space_desc["n"])
    raise ValueError(f"Unsupported space description: {space_desc}")


def _build_net(in_size: int, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_size, hidden), nn.LeakyReLU(0.01),
        nn.Linear(hidden, hidden),  nn.LeakyReLU(0.01),
        nn.Linear(hidden, 1),
    )


def traj_to_sa(trajectory: List[Tuple]) -> np.ndarray:
    """(obs, action, next_obs, r_comp, done) list → (T, obs_dim+act_dim) float32."""
    rows = []
    for obs, action, *_ in trajectory:
        rows.append(np.concatenate([np.atleast_1d(obs).ravel(),
                                    np.atleast_1d(action).ravel()]))
    return np.asarray(rows, dtype=np.float32)


class TrajectoryPreferenceBuffer:
    """Stores variable-length trajectory pairs and their preference labels.

    label = 1.0 → trajectory A preferred, 0.0 → B preferred.
    Trajectories are kept as ragged (T, sa_dim) float32 arrays.
    """

    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.A: List[np.ndarray] = []
        self.B: List[np.ndarray] = []
        self.y: List[float] = []

    def add(self, sa_a: np.ndarray, sa_b: np.ndarray, label: float) -> None:
        self.A.append(sa_a.astype(np.float32))
        self.B.append(sa_b.astype(np.float32))
        self.y.append(float(label))
        if len(self.y) > self.capacity:   # drop oldest
            self.A.pop(0); self.B.pop(0); self.y.pop(0)

    def clear(self) -> None:
        """Drop all stored pairs. The trainer calls this each round so the BT
        model is trained only on the current round's freshly-rolled-out pairs."""
        self.A.clear(); self.B.clear(); self.y.clear()

    def sample(self, batch_size: int):
        n = len(self.y)
        idxs = np.random.choice(n, size=min(batch_size, n), replace=False)
        return ([self.A[i] for i in idxs],
                [self.B[i] for i in idxs],
                np.array([self.y[i] for i in idxs], dtype=np.float32))

    def __len__(self):
        return len(self.y)


def _pad_stack(trajs: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of (T_i, D) arrays to (B, T_max, D); return (tensor, mask)
    where mask is (B, T_max, 1) with 1 for real steps, 0 for padding."""
    B = len(trajs)
    D = trajs[0].shape[1]
    T_max = max(t.shape[0] for t in trajs)
    out = np.zeros((B, T_max, D), dtype=np.float32)
    mask = np.zeros((B, T_max, 1), dtype=np.float32)
    for i, t in enumerate(trajs):
        out[i, :t.shape[0]] = t
        mask[i, :t.shape[0], 0] = 1.0
    return (torch.from_numpy(out).to(DEVICE), torch.from_numpy(mask).to(DEVICE))


class BTRewardModel:
    """Single-network Bradley-Terry reward model over full trajectories."""

    def __init__(
        self,
        env: Any,
        hidden: int = 256,
        lr: float = 3e-4,
        capacity: int = 5000,
        weight_decay: float = 1e-4,
    ):
        env_desc = describe_env(env)
        self.obs_dim = _space_dim(env_desc["observation_space"])
        self.action_dim = _space_dim(env_desc["action_space"])
        self.sa_dim = self.obs_dim + self.action_dim
        self.hidden = hidden

        self.net = _build_net(self.sa_dim, hidden).float().to(DEVICE)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self.ce_loss = nn.CrossEntropyLoss()
        self.buffer = TrajectoryPreferenceBuffer(capacity)

        # running stats for normalising predict() output when mixed into r_fixed
        self._rew_mean = 0.0
        self._rew_std = 1.0
        self._rew_count = 0

    # ── data ─────────────────────────────────────────────────
    def add_pair(self, sa_a: np.ndarray, sa_b: np.ndarray, label: float) -> None:
        self.buffer.add(sa_a, sa_b, label)

    # ── scoring ──────────────────────────────────────────────
    def _traj_scores(self, padded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, T, D), (B, T, 1) → (B, 1) summed per-step reward over real steps."""
        per_step = self.net(padded) * mask          # zero out padding
        return per_step.sum(dim=1)

    def train(self, batch_size: Optional[int] = None, n_epochs: int = 50) -> List[float]:
        """One round of BT training. Returns the per-epoch CE loss list.

        With the per-round buffer (cleared each round), the natural batch is the
        whole buffer — all C(num_trajs, 2) pairs from this round. Pass an explicit
        batch_size to mini-batch instead. Default (None) = full batch."""
        if len(self.buffer) == 0:
            return [0.0] * n_epochs
        bs = len(self.buffer) if batch_size is None else min(batch_size, len(self.buffer))
        losses: List[float] = []
        for _ in range(n_epochs):
            A, B, y = self.buffer.sample(bs)
            pa, ma = _pad_stack(A)
            pb, mb = _pad_stack(B)
            # CrossEntropyLoss target: index of the preferred trajectory.
            # label 1.0 → A preferred → index 0; label 0.0 → B → index 1.
            target = torch.from_numpy((1 - y).astype(np.int64)).to(DEVICE)
            logits = torch.cat([self._traj_scores(pa, ma),
                                self._traj_scores(pb, mb)], dim=-1)
            loss = self.ce_loss(logits, target)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            losses.append(float(loss.item()))
        self._refresh_norm_stats()
        return losses

    def accuracy(self) -> float:
        """Pairwise preference accuracy on the current buffer (in-sample)."""
        if len(self.buffer) == 0:
            return 0.0
        A, B, y = self.buffer.sample(min(256, len(self.buffer)))
        pa, ma = _pad_stack(A)
        pb, mb = _pad_stack(B)
        target = torch.from_numpy((1 - y).astype(np.int64)).to(DEVICE)
        with torch.no_grad():
            logits = torch.cat([self._traj_scores(pa, ma),
                                self._traj_scores(pb, mb)], dim=-1)
            pred = torch.argmax(logits, dim=1)
        return float((pred == target).float().mean().item())

    # ── prediction (for the Phase-II mixed reward) ───────────
    def _refresh_norm_stats(self) -> None:
        """Estimate mean/std of per-step predicted reward over a buffer sample so
        predict() can be normalised onto a comparable scale with r_fixed."""
        if len(self.buffer) == 0:
            return
        A, _, _ = self.buffer.sample(min(64, len(self.buffer)))
        sa = np.concatenate(A, axis=0)
        x = torch.from_numpy(sa).to(DEVICE)
        with torch.no_grad():
            r = self.net(x).cpu().numpy().ravel()
        self._rew_mean = float(np.mean(r))
        self._rew_std = float(np.std(r)) + 1e-6

    def predict(self, obs: np.ndarray, action: np.ndarray) -> float:
        sa = np.concatenate([np.atleast_1d(obs).ravel(),
                             np.atleast_1d(action).ravel()]).astype(np.float32)
        x = torch.from_numpy(sa).reshape(1, -1).to(DEVICE)
        with torch.no_grad():
            return float(self.net(x).item())

    def predict_normalized(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Mean-centred, std-scaled prediction for stable mixing with r_fixed."""
        return (self.predict(obs, action) - self._rew_mean) / self._rew_std

    # ── persistence ──────────────────────────────────────────
    def save(self, path: str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path / "model.pt")
        torch.save({
            "obs_dim": self.obs_dim, "action_dim": self.action_dim,
            "sa_dim": self.sa_dim, "hidden": self.hidden,
            "rew_mean": self._rew_mean, "rew_std": self._rew_std,
        }, path / "meta.pt")

    def load(self, path: str) -> None:
        path = Path(path)
        self.net.load_state_dict(torch.load(path / "model.pt", map_location=DEVICE))
        meta = torch.load(path / "meta.pt", map_location=DEVICE)
        self._rew_mean = float(meta.get("rew_mean", 0.0))
        self._rew_std = float(meta.get("rew_std", 1.0))
