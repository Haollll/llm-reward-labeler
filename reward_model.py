import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from typing import Any, List, Optional, Tuple

from llm_utils import describe_env

DEVICE = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)


def _build_net(in_size: int, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_size, hidden), nn.LeakyReLU(0.01),
        nn.Linear(hidden, hidden),  nn.LeakyReLU(0.01),
        nn.Linear(hidden, 1),       nn.Tanh(),
    )


def _space_dim(space_desc: dict) -> int:
    if "shape" in space_desc:
        dim = 1
        for size in space_desc["shape"]:
            dim *= int(size)
        return dim
    if "n" in space_desc:
        return int(space_desc["n"])
    raise ValueError(f"Unsupported space description: {space_desc}")


class PreferenceBuffer:
    """numpy circular buffer — faster than Python list"""
    def __init__(self, capacity: int, size_segment: int, sa_dim: int):
        self.capacity     = capacity
        self.size_segment = size_segment
        self.seg1  = np.zeros((capacity, size_segment, sa_dim), dtype=np.float32)
        self.seg2  = np.zeros((capacity, size_segment, sa_dim), dtype=np.float32)
        self.label = np.zeros((capacity, 1),                    dtype=np.float32)
        self.index = 0
        self.full  = False

    def add(self, seg1, seg2, label: float):
        self.seg1[self.index]  = seg1
        self.seg2[self.index]  = seg2
        self.label[self.index] = label
        self.index = (self.index + 1) % self.capacity
        if self.index == 0:
            self.full = True

    def sample(self, batch_size: int):
        max_idx = self.capacity if self.full else self.index
        idxs    = np.random.choice(max_idx, size=min(batch_size, max_idx), replace=False)
        return self.seg1[idxs], self.seg2[idxs], self.label[idxs]

    def __len__(self):
        return self.capacity if self.full else self.index


class RewardModel:
    """
    Ensemble reward model trained with Bradley-Terry loss.

    Quick usage:
        env = gym.make("HalfCheetah-v5")
        model = RewardModel(env)
        model.add(seg_A, seg_B, label=1.0)   # from compare_trajectories()
        loss = model.train()
        r    = model.predict(obs, action)     # for RewardWrappedEnv
    """
    def __init__(
        self,
        env: Any,
        ensemble_size: int = 3,
        hidden: int        = 256,
        lr: float          = 3e-4,
        size_segment: int  = 50,
        capacity: int      = 5000,
    ):
        env_desc = describe_env(env)
        obs_dim = _space_dim(env_desc["observation_space"])
        action_dim = _space_dim(env_desc["action_space"])

        self.obs_dim       = obs_dim
        self.action_dim    = action_dim
        self.sa_dim        = obs_dim + action_dim
        self.ensemble_size = ensemble_size
        self.size_segment  = size_segment
        self.hidden        = hidden

        self.nets = [
            _build_net(self.sa_dim, hidden).float().to(DEVICE)
            for _ in range(ensemble_size)
        ]
        self.optimizer = optim.Adam(
            [p for net in self.nets for p in net.parameters()], lr=lr
        )
        self.ce_loss = nn.CrossEntropyLoss()
        self.buffer  = PreferenceBuffer(capacity, size_segment, self.sa_dim)

    def add(self, seg1: np.ndarray, seg2: np.ndarray, label: float):
        """label: 1.0 = seg1 better, 0.0 = seg2 better"""
        self.buffer.add(seg1, seg2, label)

    def _sum_reward(self, seg: np.ndarray, member: int) -> torch.Tensor:
        """seg: (B, T, sa_dim) → (B, 1)"""
        return self.nets[member](torch.FloatTensor(seg).to(DEVICE)).sum(axis=1)

    def train(
        self,
        batch_size: int = 64,
        n_epochs: int = 50,
        progress_bar: bool = False,
    ) -> float:
        if len(self.buffer) < batch_size:
            return 0.0
        total = 0.0
        epochs = range(n_epochs)
        if progress_bar:
            from tqdm.auto import tqdm
            epochs = tqdm(epochs, desc="Reward model", leave=False)

        for _ in epochs:
            seg1, seg2, labels = self.buffer.sample(batch_size)
            target = torch.from_numpy(
                (1 - labels.flatten()).astype(np.int64)
            ).to(DEVICE)
            self.optimizer.zero_grad()
            loss = sum(
                self.ce_loss(
                    torch.cat([self._sum_reward(seg1, m),
                               self._sum_reward(seg2, m)], dim=-1),
                    target,
                )
                for m in range(self.ensemble_size)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for net in self.nets for p in net.parameters()], 1.0
            )
            self.optimizer.step()
            total += loss.item()
        return total / n_epochs

    def predict(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Single-step reward prediction for RewardWrappedEnv"""
        sa = np.concatenate([obs, action]).reshape(1, 1, -1).astype(np.float32)
        x  = torch.FloatTensor(sa).to(DEVICE)
        with torch.no_grad():
            r_hats = [self.nets[m](x).item() for m in range(self.ensemble_size)]
        return float(np.mean(r_hats))

    def disagreement(self, seg1: np.ndarray, seg2: np.ndarray) -> np.ndarray:
        """Ensemble disagreement score — higher = more worth asking LLM"""
        probs = []
        with torch.no_grad():
            for m in range(self.ensemble_size):
                r1 = self._sum_reward(seg1, m)
                r2 = self._sum_reward(seg2, m)
                p  = F.softmax(torch.cat([r1, r2], dim=-1), dim=-1)[:, 0]
                probs.append(p.cpu().numpy())
        return np.std(np.array(probs), axis=0)

    def select_queries(self, cands1, cands2, n: int):
        """Pick top-n most disagreed pairs to query LLM"""
        scores = self.disagreement(cands1, cands2)
        top_k  = (-scores).argsort()[:n]
        return cands1[top_k], cands2[top_k]

    def accuracy(self) -> float:
        if len(self.buffer) == 0:
            return 0.0
        seg1, seg2, labels = self.buffer.sample(min(256, len(self.buffer)))
        target = torch.from_numpy(
            (1 - labels.flatten()).astype(np.int64)
        ).to(DEVICE)
        accs = []
        with torch.no_grad():
            for m in range(self.ensemble_size):
                r    = torch.cat([self._sum_reward(seg1, m),
                                  self._sum_reward(seg2, m)], dim=-1)
                _, p = torch.max(r, dim=1)
                accs.append((p == target).float().mean().item())
        return float(np.mean(accs))

    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for i, net in enumerate(self.nets):
            torch.save(net.state_dict(), path / f"member{i}.pt")
        torch.save(
            {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "sa_dim": self.sa_dim,
                "ensemble_size": self.ensemble_size,
                "size_segment": self.size_segment,
                "hidden": self.hidden,
            },
            path / "metadata.pt",
        )

    def load(self, path: str):
        path = Path(path)
        for i, net in enumerate(self.nets):
            net.load_state_dict(torch.load(path / f"member{i}.pt", map_location=DEVICE))


def traj_to_segment(
    trajectory: List[Tuple],
    size_segment: int,
    start: Optional[int] = None,
) -> np.ndarray:
    """trajectory → (size_segment, obs_dim + action_dim)"""
    if start is None:
        max_start = max(0, len(trajectory) - size_segment)
        start     = np.random.randint(0, max_start + 1)
    segment = []
    for i in range(start, min(start + size_segment, len(trajectory))):
        obs, action, *_ = trajectory[i]
        segment.append(np.concatenate([obs, action]))
    while len(segment) < size_segment:
        segment.append(segment[-1])
    return np.array(segment, dtype=np.float32)


if __name__ == "__main__":
    import gymnasium as gym

    print(f"Device: {DEVICE}")
    env = gym.make("HalfCheetah-v5")
    model = RewardModel(env)
    for _ in range(80):
        s1 = np.random.randn(50, 23).astype(np.float32)
        s2 = np.random.randn(50, 23).astype(np.float32)
        model.add(s1, s2, float(np.random.randint(0, 2)))
    loss = model.train(batch_size=32, n_epochs=10)
    print(f"loss={loss:.4f} | acc={model.accuracy():.2%} | buffer={len(model.buffer)}")
    r = model.predict(np.random.randn(17), np.random.randn(6))
    print(f"predicted reward: {r:.4f}")
    env.close()
