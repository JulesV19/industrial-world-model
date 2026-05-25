import os
import glob
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

HISTORY_K = 100


# ── Normalizer pour les offsets coins ─────────────────────────────────────────
class Normalizer:
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
    def __init__(self):
        self.mean = None
        self.std  = None

    def fit(self, trajs: list[np.ndarray]):
        all_data  = np.concatenate(trajs, axis=0)
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


# ---------------------------------------------------------------------------
def _is_session_file(path: str) -> bool:
    return bool(re.match(r".*session_\d+_piece\d+\.npz$", path))


def _session_id(path: str) -> str:
    return re.search(r"session_(\d+)_piece", os.path.basename(path)).group(1)


# ── Dataset ───────────────────────────────────────────────────────────────────
class DrillDataset(Dataset):
    """
    Accepte fichiers session (session_SSS_pieceNNNN.npz) et legacy (episode_NNN.npz).
    Nouveaux champs : deviation_history (K,), piece_count, cadence.
    """

    def __init__(
        self,
        episode_paths:   list[str],
        normalizer:      Normalizer     | None = None,
        traj_normalizer: TrajNormalizer | None = None,
    ):
        # Charger les tableaux de déviations/erreurs par session
        session_devs: dict[str, np.ndarray] = {}
        session_cads: dict[str, float] = {}
        for sp in episode_paths:
            if _is_session_file(sp):
                sid = _session_id(sp)
                if sid not in session_devs:
                    data_dir = os.path.dirname(sp)
                    dev_path = os.path.join(data_dir, f"session_{sid}_deviations.npy")
                    cad_path = os.path.join(data_dir, f"session_{sid}_cadence.npy")
                    session_devs[sid] = np.load(dev_path) if os.path.exists(dev_path) \
                                        else np.zeros(1000, dtype=np.float32)
                    session_cads[sid] = float(np.load(cad_path)) if os.path.exists(cad_path) else 0.0

        corners_list = []
        speeds       = []
        offsets_list = []
        defects_list = []
        trajs_list   = []
        hist_list    = []
        pc_list      = []
        cad_list     = []

        for ep_path in sorted(episode_paths):
            data    = np.load(ep_path)
            corners = data["corner_targets"].astype(np.float32)
            hits    = data["drill_hits"].astype(np.float32)
            speed   = float(data["duration_per_segment"])
            defects = data["defects"].astype(np.float32)
            q_real  = data["q_real"].astype(np.float32)
            pc      = int(data["piece_count"])   if "piece_count" in data else 0
            cad     = float(data["cadence"])     if "cadence"     in data else 0.0

            if _is_session_file(ep_path):
                sid   = _session_id(ep_path)
                n     = pc
                devs  = session_devs[sid]
                start = max(0, n - HISTORY_K)
                hist  = devs[start:n]
                pad   = np.zeros(HISTORY_K - len(hist), dtype=np.float32)
                history = np.concatenate([pad, hist])
            else:
                history = np.zeros(HISTORY_K, dtype=np.float32)

            corners_list.append(corners)
            speeds.append(speed)
            offsets_list.append(hits - corners)
            defects_list.append(defects)
            trajs_list.append(q_real)
            hist_list.append(history)
            pc_list.append(np.float32(pc))
            cad_list.append(np.float32(cad))

        offsets_arr = np.stack(offsets_list)

        if normalizer is None:
            normalizer = Normalizer()
            normalizer.fit(offsets_arr)
        self.normalizer = normalizer

        if traj_normalizer is None:
            traj_normalizer = TrajNormalizer()
            traj_normalizer.fit(trajs_list)
        self.traj_normalizer = traj_normalizer

        self.samples = []
        for corners, speed, offsets, defects, q_real, history, pc, cad in zip(
            corners_list, speeds, offsets_arr, defects_list, trajs_list,
            hist_list, pc_list, cad_list
        ):
            self.samples.append({
                "corners":           corners,
                "speed":             np.float32(speed),
                "q_real_norm":       traj_normalizer.normalize(q_real),
                "length":            len(q_real),
                "offsets_norm":      normalizer.normalize(offsets),
                "defects":           defects,
                "deviation_history": history,
                "piece_count":       pc,
                "cadence":           cad,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.from_numpy(s["corners"]),
            torch.tensor(s["speed"]),
            torch.from_numpy(s["q_real_norm"]),
            s["length"],
            torch.from_numpy(s["offsets_norm"]),
            torch.from_numpy(s["defects"]),
            torch.from_numpy(s["deviation_history"]),   # (K,)
            torch.tensor(s["piece_count"]),
            torch.tensor(s["cadence"]),
        )


def collate_fn(batch):
    (corners, speeds, trajs, lengths,
     offsets, defects, hist_list, pc_list, cad_list) = zip(*batch)
    return (
        torch.stack(corners),
        torch.stack(speeds),
        pad_sequence(trajs, batch_first=True, padding_value=0.0),
        torch.tensor(lengths),
        torch.stack(offsets),
        torch.stack(defects),
        torch.stack(hist_list).unsqueeze(-1),      # (B, K, 1)
        torch.stack(pc_list).unsqueeze(-1),        # (B, 1)
        torch.stack(cad_list).unsqueeze(-1),       # (B, 1)
    )
