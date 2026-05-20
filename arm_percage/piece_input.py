"""
Extrait les 4 coins réels d'une pièce depuis un épisode de découpe.

Stratégie :
  1. Les coins idéaux sont lus dans pieces_database.json (waypoints is_cutting=True)
  2. Pour chaque coin, on trouve le pas de temps où fk(q_des) est le plus proche
  3. On lit q_real à ce pas de temps → position réelle du laser lors du passage
  4. Un petit bruit de placement simule l'incertitude du convoyeur (modélisé plus tard)
"""

import json
import os
import re

import numpy as np

from config import l1, l2

# Incertitude de placement de la pièce au poste de perçage (convoyeur).
# Remplacé par le vrai modèle convoyeur quand il sera prêt.
PLACEMENT_NOISE_STD = 0.002   # 2 mm (écart-type, mètres)
PLACEMENT_NOISE_SEED = None   # None = non-déterministe

# Recul des cibles de perçage vers l'intérieur de la pièce.
DRILL_INSET = 0.1            # 5 cm depuis chaque coin vers le centre


def _fk(q: np.ndarray) -> np.ndarray:
    """q : (2,) ou (T, 2) → (2,) ou (T, 2)"""
    if q.ndim == 1:
        x = l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1])
        y = l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])
        return np.array([x, y])
    x = l1 * np.cos(q[:, 0]) + l2 * np.cos(q[:, 0] + q[:, 1])
    y = l1 * np.sin(q[:, 0]) + l2 * np.sin(q[:, 0] + q[:, 1])
    return np.stack([x, y], axis=1)


def _ideal_corners_from_db(pieces_db_path: str, piece_idx: int) -> np.ndarray:
    """
    Retourne les 4 coins idéaux (x, y) du carré, dans l'ordre de découpe.
    Les coins = waypoints avec is_cutting=True, dédupliqués (le 4e referme la boucle).
    """
    with open(pieces_db_path) as f:
        db = json.load(f)
    waypoints = db["pieces"][piece_idx]   # liste de [x, y, is_cutting]

    seen = []
    for wp in waypoints:
        if wp[2]:  # is_cutting
            pt = (round(wp[0], 9), round(wp[1], 9))
            if pt not in seen:
                seen.append(pt)

    assert len(seen) == 4, f"Attendu 4 coins distincts, trouvé {len(seen)}"
    return np.array(seen, dtype=np.float64)  # (4, 2)


def _piece_idx_from_path(npz_path: str) -> int:
    """Déduit l'index de pièce depuis le nom de fichier 'episode_001_run00.npz'."""
    basename = os.path.basename(npz_path)
    m = re.match(r"episode_(\d+)_run\d+\.npz", basename)
    assert m, f"Nom de fichier inattendu : {basename}"
    return int(m.group(1)) - 1  # 1-indexé dans le nom, 0-indexé dans la DB


def extract_corners(npz_path: str,
                    pieces_db_path: str,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Retourne les 4 coins tels que le bras de perçage les "voit" : position
    réelle de l'outil de découpe + bruit de placement du convoyeur.

    Paramètres
    ----------
    npz_path       : chemin vers l'épisode de découpe (.npz)
    pieces_db_path : chemin vers pieces_database.json
    rng            : générateur numpy pour la reproductibilité (optionnel)

    Retourne
    --------
    corners : (4, 2) float64 — positions cartésiennes perçues des 4 coins
    """
    data = np.load(npz_path)
    q_des  = data["q_des"].astype(np.float64)   # (T, 2)
    q_real = data["q_real"].astype(np.float64)  # (T, 2)

    piece_idx    = _piece_idx_from_path(npz_path)
    ideal_corners = _ideal_corners_from_db(pieces_db_path, piece_idx)  # (4, 2)

    ee_des = _fk(q_des)  # (T, 2) — trajectoire désirée en cartésien

    real_corners = np.zeros((4, 2))
    for i, corner in enumerate(ideal_corners):
        dists = np.linalg.norm(ee_des - corner, axis=1)
        t_closest = int(np.argmin(dists))
        real_corners[i] = _fk(q_real[t_closest])

    # Décalage vers l'intérieur : chaque coin recule de DRILL_INSET vers le centre
    center = real_corners.mean(axis=0)
    for i in range(4):
        direction = center - real_corners[i]
        norm = np.linalg.norm(direction)
        if norm > 1e-9:
            real_corners[i] += (direction / norm) * DRILL_INSET

    # Bruit de placement convoyeur
    if rng is None:
        rng = np.random.default_rng(PLACEMENT_NOISE_SEED)
    noise = rng.normal(0.0, PLACEMENT_NOISE_STD, size=(4, 2))
    real_corners += noise

    return real_corners  # (4, 2)
