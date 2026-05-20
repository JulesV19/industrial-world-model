import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


# ── Normalizer pour les offsets coins ─────────────────────────────────────────
class Normalizer:
    """Z-score sur les offsets de perçage (drill_hit - corner_target)."""

    def __init__(self):
        self.mean = None
        self.std  = None

    def fit(self, offsets: np.ndarray):
        """offsets : (N, 4, 2)"""
        flat = offsets.reshape(-1, 2)
        self.mean = flat.mean(axis=0).astype(np.float32)
        self.std  = (flat.std(axis=0) + 1e-6).astype(np.float32)

    def normalize(self, offsets: np.ndarray) -> np.ndarray:
        return ((offsets - self.mean) / self.std).astype(np.float32)

    def denormalize(self, offsets: np.ndarray) -> np.ndarray:
        return (offsets * self.std + self.mean).astype(np.float32)

    def denormalize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
        std  = torch.tensor(self.std,  dtype=x.dtype, device=x.device)
        return x * std + mean

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        n = cls()
        d = np.load(path)
        n.mean = d["mean"].astype(np.float32)
        n.std  = d["std"].astype(np.float32)
        return n


# ── Normalizer pour la trajectoire q_real ─────────────────────────────────────
class TrajNormalizer:
    """Z-score sur les angles articulaires q_real (T, 2)."""

    def __init__(self):
        self.mean = None
        self.std  = None

    def fit(self, trajs: list[np.ndarray]):
        """trajs : liste de (T_i, 2)"""
        all_data  = np.concatenate(trajs, axis=0)          # (sum_T, 2)
        self.mean = all_data.mean(axis=0).astype(np.float32)
        self.std  = (all_data.std(axis=0) + 1e-6).astype(np.float32)

    def normalize(self, traj: np.ndarray) -> np.ndarray:
        return ((traj - self.mean) / self.std).astype(np.float32)

    def denormalize(self, traj: np.ndarray) -> np.ndarray:
        return (traj * self.std + self.mean).astype(np.float32)

    def denormalize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
        std  = torch.tensor(self.std,  dtype=x.dtype, device=x.device)
        return x * std + mean

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str) -> "TrajNormalizer":
        n = cls()
        d = np.load(path)
        n.mean = d["mean"].astype(np.float32)
        n.std  = d["std"].astype(np.float32)
        return n


# ── Dataset ───────────────────────────────────────────────────────────────────
class DrillDataset(Dataset):
    """
    Un sample = un épisode de perçage.

    Entrée  : corner_targets (4, 2) + speed (1,)
    Sortie  : q_real normalisée (T, 2), longueur T,
              offset normalisé (4, 2), defects (4,)
    """

    def __init__(
        self,
        episode_paths:   list[str],
        normalizer:      Normalizer     | None = None,
        traj_normalizer: TrajNormalizer | None = None,
    ):
        corners_list = []
        speeds       = []
        offsets_list = []
        defects_list = []
        trajs_list   = []

        for ep_path in sorted(episode_paths):
            data    = np.load(ep_path)
            corners = data["corner_targets"].astype(np.float32)  # (4, 2)
            hits    = data["drill_hits"].astype(np.float32)      # (4, 2)
            speed   = float(data["duration_per_segment"])
            defects = data["defects"].astype(np.float32)         # (4,)
            q_real  = data["q_real"].astype(np.float32)          # (T, 2)

            corners_list.append(corners)
            speeds.append(speed)
            offsets_list.append(hits - corners)
            defects_list.append(defects)
            trajs_list.append(q_real)

        offsets_arr = np.stack(offsets_list)   # (N, 4, 2)

        if normalizer is None:
            normalizer = Normalizer()
            normalizer.fit(offsets_arr)
        self.normalizer = normalizer

        if traj_normalizer is None:
            traj_normalizer = TrajNormalizer()
            traj_normalizer.fit(trajs_list)
        self.traj_normalizer = traj_normalizer

        self.samples = []
        for corners, speed, offsets, defects, q_real in zip(
            corners_list, speeds, offsets_arr, defects_list, trajs_list
        ):
            self.samples.append({
                "corners":      corners,
                "speed":        np.float32(speed),
                "q_real_norm":  traj_normalizer.normalize(q_real),  # (T, 2)
                "length":       len(q_real),
                "offsets_norm": normalizer.normalize(offsets),       # (4, 2)
                "defects":      defects,                             # (4,)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.from_numpy(s["corners"]),       # (4, 2)
            torch.tensor(s["speed"]),             # ()
            torch.from_numpy(s["q_real_norm"]),   # (T, 2)
            s["length"],                          # int
            torch.from_numpy(s["offsets_norm"]),  # (4, 2)
            torch.from_numpy(s["defects"]),       # (4,)
        )


def collate_fn(batch):
    corners, speeds, trajs, lengths, offsets, defects = zip(*batch)
    return (
        torch.stack(corners),                                              # (B, 4, 2)
        torch.stack(speeds),                                               # (B,)
        pad_sequence(trajs, batch_first=True, padding_value=0.0),         # (B, T_max, 2)
        torch.tensor(lengths),                                             # (B,)
        torch.stack(offsets),                                              # (B, 4, 2)
        torch.stack(defects),                                              # (B, 4)
    )
