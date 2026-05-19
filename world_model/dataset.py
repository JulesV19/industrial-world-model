import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# Tous les signaux disponibles et leur nombre de colonnes
METRIC_KEYS = [
    ("q_real",     2),
    ("q_sensed",   2),
    ("q_des",      2),
    ("dq_real",    2),
    ("dq_sensed",  2),
    ("dq_des",     2),
    ("tau",        2),
    ("is_cutting", 1),
]
# Clés calculées (non stockées dans le npz)
COMPUTED_KEYS = {
    "q_error": lambda data: data["q_real"] - data["q_des"],  # erreur réelle de la machine
}

OBS_DIM = sum(d for _, d in METRIC_KEYS)   # 15

CONTINUOUS_SLICE = slice(0, 14)
BINARY_IDX       = 14


def build_obs_vector(data, keys=None) -> np.ndarray:
    """
    Concatène les signaux demandés en un vecteur (T, D).
    keys : liste de noms de signaux, ex. ["q_des"] ou None pour tous.
           Accepte aussi les clés calculées comme "q_error".
    """
    if keys is None:
        keys = [k for k, _ in METRIC_KEYS]
    parts = []
    for key in keys:
        if key in COMPUTED_KEYS:
            arr = COMPUTED_KEYS[key](data)
        else:
            arr = data[key]
        if arr.ndim == 1:
            arr = arr[:, None]
        parts.append(arr)
    return np.concatenate(parts, axis=-1).astype(np.float32)


COMPUTED_KEY_DIMS = {"q_error": 2}

def target_dim(keys) -> int:
    """Dimension totale des signaux cibles."""
    d = {**dict(METRIC_KEYS), **COMPUTED_KEY_DIMS}
    return sum(d[k] for k in keys)


# ---------------------------------------------------------------------------
class Normalizer:
    """Z-score normalizer. is_cutting laissé en 0/1 si présent."""

    def __init__(self):
        self.mean = None
        self.std  = None
        self.keys = None   # signaux pour lesquels le normalizer a été fitté

    def fit(self, trajectories: list[np.ndarray], keys=None):
        self.keys = keys
        all_data  = np.concatenate(trajectories, axis=0)
        self.mean = all_data.mean(axis=0).astype(np.float32)
        self.std  = (all_data.std(axis=0) + 1e-6).astype(np.float32)
        # Ne pas normaliser is_cutting
        if keys and "is_cutting" in keys:
            idx = keys.index("is_cutting")
            self.mean[idx] = 0.0
            self.std [idx] = 1.0
        elif keys is None:
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
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std,
                 keys=np.array(self.keys or [], dtype=str))

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        n = cls()
        d = np.load(path, allow_pickle=True)
        n.mean = d["mean"].astype(np.float32)
        n.std  = d["std"] .astype(np.float32)
        k = d["keys"].tolist()
        n.keys = k if k else None
        return n


# ---------------------------------------------------------------------------
class TrajectoryDataset(Dataset):
    """
    target_keys : signaux à prédire, ex. ["q_des"] ou None pour tous les 15.
    """

    def __init__(
        self,
        episode_paths: list[str],
        piece_db: list,
        normalizer: Normalizer | None = None,
        target_keys: list[str] | None = None,
    ):
        self.target_keys = target_keys   # None → tous les signaux

        raw_trajs = []
        meta      = []

        for ep_path in sorted(episode_paths):
            idx       = int(os.path.basename(ep_path).split("_")[1].split(".")[0]) - 1
            data      = np.load(ep_path)
            obs       = build_obs_vector(data, target_keys)
            waypoints = np.array(piece_db[idx], dtype=np.float32)
            speed     = float(data["duration_per_segment"])
            raw_trajs.append(obs)
            meta.append((waypoints, speed, len(obs)))

        if normalizer is None:
            normalizer = Normalizer()
            normalizer.fit(raw_trajs, keys=target_keys)
        self.normalizer = normalizer

        self.samples = []
        for (waypoints, speed, seq_len), obs in zip(meta, raw_trajs):
            self.samples.append({
                "waypoints": waypoints,
                "speed":     np.float32(speed),
                "obs":       normalizer.normalize(obs),
                "length":    seq_len,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.from_numpy(s["waypoints"]),
            torch.tensor(s["speed"]),
            torch.from_numpy(s["obs"]),
            s["length"],
        )


# ---------------------------------------------------------------------------
def collate_fn(batch):
    waypoints_list, speeds, obs_list, lengths = zip(*batch)
    return (
        pad_sequence(waypoints_list, batch_first=True, padding_value=0.0),
        torch.tensor([w.shape[0] for w in waypoints_list]),
        torch.stack(speeds).unsqueeze(-1),          # (B, 1)
        pad_sequence(obs_list, batch_first=True, padding_value=0.0),
        torch.tensor(lengths),
    )
