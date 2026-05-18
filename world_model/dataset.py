import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# Ordered list of all recorded signals and their column widths
METRIC_KEYS = [
    ("q_real",    2),
    ("q_sensed",  2),
    ("q_des",     2),
    ("dq_real",   2),
    ("dq_sensed", 2),
    ("dq_des",    2),
    ("tau",       2),
    ("is_cutting", 1),
]
OBS_DIM = sum(d for _, d in METRIC_KEYS)   # 15

# Indices of the continuous signals (MSE loss) vs binary (BCE loss)
CONTINUOUS_SLICE = slice(0, 14)   # q_real … tau
BINARY_IDX       = 14             # is_cutting


def build_obs_vector(data) -> np.ndarray:
    """Concatenate all metrics from a loaded .npz file into (T, OBS_DIM)."""
    parts = []
    for key, dim in METRIC_KEYS:
        arr = data[key]
        if arr.ndim == 1:
            arr = arr[:, None]
        parts.append(arr)
    return np.concatenate(parts, axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
class Normalizer:
    """Z-score normalizer fitted on training data (is_cutting left untouched)."""

    def __init__(self):
        self.mean: np.ndarray | None = None
        self.std:  np.ndarray | None = None

    def fit(self, trajectories: list[np.ndarray]):
        all_data = np.concatenate(trajectories, axis=0)   # (N_total, OBS_DIM)
        self.mean = all_data.mean(axis=0).astype(np.float32)
        self.std  = (all_data.std(axis=0) + 1e-6).astype(np.float32)
        # Keep is_cutting as 0/1 — no normalisation
        self.mean[BINARY_IDX] = 0.0
        self.std [BINARY_IDX] = 1.0

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std + self.mean).astype(np.float32)

    def denormalize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
        std  = torch.tensor(self.std,  dtype=x.dtype, device=x.device)
        return x * std + mean

    def save(self, path: str):
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        n = cls()
        d = np.load(path)
        n.mean = d["mean"].astype(np.float32)
        n.std  = d["std"] .astype(np.float32)
        return n


# ---------------------------------------------------------------------------
class TrajectoryDataset(Dataset):
    """
    One sample = (waypoints, normalized_trajectory, seq_length).

    Episode k  ↔  piece index k-1  (filename: episode_001.npz → piece 0).
    """

    def __init__(
        self,
        episode_paths: list[str],
        piece_db: list,
        normalizer: Normalizer | None = None,
    ):
        raw_trajs = []
        meta      = []

        for ep_path in sorted(episode_paths):
            idx = int(os.path.basename(ep_path).split("_")[1].split(".")[0]) - 1
            data = np.load(ep_path)
            obs  = build_obs_vector(data)              # (T, 15)
            waypoints = np.array(piece_db[idx], dtype=np.float32)  # (N_wp, 3)
            raw_trajs.append(obs)
            meta.append((waypoints, len(obs)))

        if normalizer is None:
            normalizer = Normalizer()
            normalizer.fit(raw_trajs)
        self.normalizer = normalizer

        self.samples = []
        for (waypoints, seq_len), obs in zip(meta, raw_trajs):
            obs_norm = normalizer.normalize(obs)
            self.samples.append({
                "waypoints": waypoints,
                "obs":       obs_norm,
                "length":    seq_len,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.from_numpy(s["waypoints"]),   # (N_wp, 3)
            torch.from_numpy(s["obs"]),          # (T, 15)
            s["length"],                          # int
        )


# ---------------------------------------------------------------------------
def collate_fn(batch):
    """Pad waypoints and trajectories to the max length in the batch."""
    waypoints_list, obs_list, lengths = zip(*batch)

    wp_lengths  = torch.tensor([w.shape[0] for w in waypoints_list])
    seq_lengths = torch.tensor(lengths)

    waypoints_padded = pad_sequence(waypoints_list, batch_first=True, padding_value=0.0)
    obs_padded       = pad_sequence(obs_list,       batch_first=True, padding_value=0.0)

    return waypoints_padded, wp_lengths, obs_padded, seq_lengths
