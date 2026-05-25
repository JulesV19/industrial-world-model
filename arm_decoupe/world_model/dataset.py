import os
import glob
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

HISTORY_K   = 100
HISTORY_DIM = 3    # [mean_cut_deviation, tau_cut_rms, q_error_cut_rms]

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
COMPUTED_KEYS = {
    "q_error": lambda data: data["q_real"] - data["q_des"],
}

OBS_DIM = sum(d for _, d in METRIC_KEYS)   # 15

CONTINUOUS_SLICE = slice(0, 14)
BINARY_IDX       = 14


def build_obs_vector(data, keys=None) -> np.ndarray:
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
    d = {**dict(METRIC_KEYS), **COMPUTED_KEY_DIMS}
    return sum(d[k] for k in keys)


# ---------------------------------------------------------------------------
class Normalizer:
    def __init__(self):
        self.mean = None
        self.std  = None
        self.keys = None

    def fit(self, trajectories: list[np.ndarray], keys=None):
        self.keys = keys
        all_data  = np.concatenate(trajectories, axis=0)
        self.mean = all_data.mean(axis=0).astype(np.float32)
        self.std  = (all_data.std(axis=0) + 1e-6).astype(np.float32)
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
def _load_session_history(data_dir: str, sid: str) -> np.ndarray | None:
    """
    Charge l'historique (N, 3) d'une session : [deviation, tau_cut_rms, q_error_cut_rms].
    Retourne None si introuvable (anciens datasets sans le fichier).
    """
    path = os.path.join(data_dir, f"session_{sid}_history.npy")
    if os.path.exists(path):
        return np.load(path).astype(np.float32)
    # Rétrocompatibilité : anciens datasets avec seulement deviations.npy
    dev_path = os.path.join(data_dir, f"session_{sid}_deviations.npy")
    if os.path.exists(dev_path):
        devs = np.load(dev_path).astype(np.float32)
        return np.stack([devs, np.zeros_like(devs), np.zeros_like(devs)], axis=1)
    return None


def _build_history(session_history: np.ndarray | None, piece_count: int) -> np.ndarray:
    """
    Construit le vecteur d'historique (K, 3) depuis le tableau (N, 3) de la session.
    Si session_history est None, retourne des zéros.
    """
    if session_history is None:
        return np.zeros((HISTORY_K, HISTORY_DIM), dtype=np.float32)
    n     = int(piece_count)
    start = max(0, n - HISTORY_K)
    hist  = session_history[start:n]                                   # (<= K, 3)
    pad   = np.zeros((HISTORY_K - len(hist), HISTORY_DIM), dtype=np.float32)
    return np.concatenate([pad, hist], axis=0)                         # (K, 3)


def _load_episode(ep_path: str, piece_db: list | None,
                  target_keys: list[str] | None) -> dict:
    """
    Charge un épisode et retourne un dict avec waypoints, speed, obs, quality,
    piece_count, cadence, mean_cut_deviation.
    Compatible ancien format (episode_PPP_runRR.npz) et nouveau (session_SSS_pieceNNNN.npz).
    """
    data = np.load(ep_path)

    # Waypoints : stockés dans le npz (nouveau format) ou extraits de piece_db (ancien)
    if "waypoints" in data:
        waypoints = data["waypoints"].astype(np.float32)
    else:
        fname = os.path.basename(ep_path)
        idx   = int(fname.split("_")[1]) - 1
        waypoints = np.array(piece_db[idx], dtype=np.float32)

    obs     = build_obs_vector(data, target_keys)
    speed   = float(data["duration_per_segment"])

    mean_dev = float(data["mean_cut_deviation"]) if "mean_cut_deviation" in data else 0.0
    pc       = int(data["piece_count"])           if "piece_count"        in data else 0
    cad      = float(data["cadence"])             if "cadence"            in data else 0.0

    return dict(
        waypoints           = waypoints,
        speed               = np.float32(speed),
        obs                 = obs,
        length              = len(obs),
        cut_deviation       = data["cut_deviation"].astype(np.float32),
        cut_defect          = data["cut_defect"].astype(np.float32),
        is_cutting          = data["is_cutting"].astype(np.float32),
        mean_cut_deviation  = np.float32(mean_dev),
        piece_count         = pc,
        cadence             = np.float32(cad),
        temperature         = np.float32(data["temperature"]) if "temperature" in data else np.float32(20.0),
    )


def _is_session_file(path: str) -> bool:
    return bool(re.match(r".*session_\d+_piece\d+\.npz$", path))


def _session_id(path: str) -> str:
    """Extrait 'SSS' depuis session_SSS_pieceNNNN.npz."""
    return re.search(r"session_(\d+)_piece", os.path.basename(path)).group(1)


def split_by_session(episode_paths: list[str], val_frac: float = 0.1,
                     seed: int = 42) -> tuple[list[str], list[str]]:
    """
    Sépare les chemins par session complète : toutes les pièces d'une même session
    vont dans le même split (train ou val), ce qui évite la fuite d'information
    via les historiques de déviation.
    Les fichiers legacy (episode_PPP_runRR.npz) sont séparés au niveau épisode.
    Retourne (train_paths, val_paths).
    """
    session_files = [p for p in episode_paths if _is_session_file(p)]
    legacy_files  = [p for p in episode_paths if not _is_session_file(p)]

    rng = np.random.default_rng(seed)

    sessions: dict[str, list[str]] = {}
    for p in session_files:
        sessions.setdefault(_session_id(p), []).append(p)

    sids = sorted(sessions.keys())
    if sids:
        perm   = rng.permutation(len(sids))
        n_val  = max(1, int(len(sids) * val_frac))
        val_sids   = {sids[i] for i in perm[:n_val]}
        train_sids = {sids[i] for i in perm[n_val:]}
    else:
        val_sids, train_sids = set(), set()

    train_paths = [p for sid in train_sids for p in sessions[sid]]
    val_paths   = [p for sid in val_sids   for p in sessions[sid]]

    if legacy_files:
        perm_leg  = rng.permutation(len(legacy_files))
        n_val_leg = max(0, int(len(legacy_files) * val_frac))
        val_paths   += [legacy_files[i] for i in perm_leg[:n_val_leg]]
        train_paths += [legacy_files[i] for i in perm_leg[n_val_leg:]]

    return sorted(train_paths), sorted(val_paths)


# ---------------------------------------------------------------------------
class TrajectoryDataset(Dataset):
    """
    Accepte deux formats de fichiers :
    - Ancien : episode_PPP_runRR.npz  → history=zeros, piece_count=0, cadence=0
    - Nouveau : session_SSS_pieceNNNN.npz + session_SSS_deviations.npy
                → history = K déviations précédentes de la session

    target_keys : signaux à prédire, ex. ["q_des"] ou None pour tous.
    """

    def __init__(
        self,
        episode_paths: list[str],
        piece_db: list | None = None,
        normalizer: Normalizer | None = None,
        target_keys: list[str] | None = None,
    ):
        self.target_keys = target_keys

        # Séparer les fichiers session des fichiers legacy
        session_paths = [p for p in episode_paths if _is_session_file(p)]
        legacy_paths  = [p for p in episode_paths if not _is_session_file(p)]

        # Charger les historiques par session (N, 3) : [deviation, tau_rms, q_error_rms]
        session_hists: dict[str, np.ndarray | None] = {}
        session_cads:  dict[str, float] = {}
        for sp in session_paths:
            sid = _session_id(sp)
            if sid not in session_hists:
                data_dir = os.path.dirname(sp)
                cad_path = os.path.join(data_dir, f"session_{sid}_cadence.npy")
                session_hists[sid] = _load_session_history(data_dir, sid)
                session_cads[sid]  = float(np.load(cad_path)) if os.path.exists(cad_path) else 0.0

        raw_trajs = []
        items     = []

        for ep_path in sorted(episode_paths):
            ep = _load_episode(ep_path, piece_db, target_keys)
            raw_trajs.append(ep["obs"])

            if _is_session_file(ep_path):
                sid     = _session_id(ep_path)
                n       = ep["piece_count"]
                history = _build_history(session_hists[sid], n)  # (K, 3)
            else:
                history = np.zeros((HISTORY_K, HISTORY_DIM), dtype=np.float32)

            items.append((ep, history))

        if normalizer is None:
            normalizer = Normalizer()
            normalizer.fit(raw_trajs, keys=target_keys)
        self.normalizer = normalizer

        all_dev = np.concatenate([
            ep["cut_deviation"][ep["is_cutting"] > 0.5]
            for ep, _ in items
            if (ep["is_cutting"] > 0.5).any()
        ])
        self.dev_mean = float(all_dev.mean())
        self.dev_std  = float(all_dev.std() + 1e-6)

        self.samples = []
        for ep, history in items:
            dev_norm = (ep["cut_deviation"] - self.dev_mean) / self.dev_std
            self.samples.append({
                "waypoints":           ep["waypoints"],
                "speed":               ep["speed"],
                "obs":                 normalizer.normalize(ep["obs"]),
                "length":              ep["length"],
                "cut_deviation":       dev_norm.astype(np.float32),
                "cut_defect":          ep["cut_defect"],
                "is_cutting":          ep["is_cutting"],
                "deviation_history":   history.astype(np.float32),  # (K, 3)
                "piece_count":         np.float32(ep["piece_count"]),
                "cadence":             np.float32(ep["cadence"]),
                "temperature":         np.float32(ep["temperature"]),
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
            torch.from_numpy(s["cut_deviation"]),
            torch.from_numpy(s["cut_defect"]),
            torch.from_numpy(s["is_cutting"]),
            torch.from_numpy(s["deviation_history"]),   # (K, 3)
            torch.tensor(s["piece_count"]),
            torch.tensor(s["cadence"]),
            torch.tensor(s["temperature"]),
        )


# ---------------------------------------------------------------------------
def collate_fn(batch):
    (waypoints_list, speeds, obs_list, lengths,
     dev_list, defect_list, cut_list,
     hist_list, pc_list, cad_list, temp_list) = zip(*batch)

    return (
        pad_sequence(waypoints_list, batch_first=True, padding_value=0.0),
        torch.tensor([w.shape[0] for w in waypoints_list]),
        torch.stack(speeds).unsqueeze(-1),                         # (B, 1)
        pad_sequence(obs_list,    batch_first=True, padding_value=0.0),
        torch.tensor(lengths),
        pad_sequence(dev_list,    batch_first=True, padding_value=0.0),
        pad_sequence(defect_list, batch_first=True, padding_value=0.0),
        pad_sequence(cut_list,    batch_first=True, padding_value=0.0),
        torch.stack(hist_list),                                     # (B, K, 3)
        torch.stack(pc_list).unsqueeze(-1),                        # (B, 1)
        torch.stack(cad_list).unsqueeze(-1),                       # (B, 1)
        torch.stack(temp_list).unsqueeze(-1),                      # (B, 1)
    )
