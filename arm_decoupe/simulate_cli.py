"""
CLI runner — simulation de découpe via le world model.

Usage:
  python simulate_cli.py --waypoints '[...]' --speeds '[0.5, 1.0]'

Retourne JSON :
  [{"speed": 1.0, "defect_pct": 12.3, "mean_deviation_mm": 8.1,
    "max_deviation_mm": 31.4, "real_corners": [[x,y], ...]}, ...]
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arm import TrajectoryPlanner
from config import l1, l2
from world_model.model import WorldModel
from world_model.dataset import Normalizer

CKPT_DIR           = os.path.join(os.path.dirname(__file__), "world_model", "checkpoints")
CKPT_PATH          = os.path.join(CKPT_DIR, "best_model.pt")
NORM_PATH          = os.path.join(CKPT_DIR, "normalizer.npz")
CUT_DEFECT_THR_M   = 0.02    # 20 mm
DRILL_INSET        = 0.1     # retraite vers le centre pour les trous (m)
PLACEMENT_NOISE_SD = 0.002   # bruit placement convoyeur (m)
DT                 = 0.01    # pas physique : 100 Hz
LOG_SUBSAMPLE      = 10      # 100 Hz → 10 Hz dans le dataset


# ── FK helpers ────────────────────────────────────────────────────────────────

def _fk(q: np.ndarray) -> np.ndarray:
    """q : (T, 2) → (T, 2) positions cartésiennes."""
    x = l1 * np.cos(q[:, 0]) + l2 * np.cos(q[:, 0] + q[:, 1])
    y = l1 * np.sin(q[:, 0]) + l2 * np.sin(q[:, 0] + q[:, 1])
    return np.stack([x, y], axis=1)


# ── Planner-only q_des generation (no physics) ───────────────────────────────

def _generate_q_des(waypoints: list, speed: float):
    """
    Simule le cycle usine (ARRIVING → CUTTING → EVACUATING) avec uniquement
    le planificateur de trajectoire (pas de physique).
    Retourne q_des (T, 2) et is_cutting (T,) à la fréquence de log (10 Hz).
    """
    planner          = TrajectoryPlanner(waypoints, duration_per_segment=speed)
    factory_state    = "ARRIVING"
    conveyor_offset  = -800.0

    q_des_list, ic_list = [], []
    step, done          = 0, False

    while not done:
        if factory_state == "ARRIVING":
            q_des = planner.last_q.copy()
            ic    = 0.0
            conveyor_offset += 500.0 * DT
            if conveyor_offset >= 0.0:
                factory_state   = "CUTTING"
                conveyor_offset = 0.0

        elif factory_state == "CUTTING":
            q_des, _, _ = planner.get_desired_state(DT)
            ic = 1.0 if planner.is_cutting else 0.0
            if planner.done:
                factory_state = "EVACUATING"

        elif factory_state == "EVACUATING":
            q_des, _, _ = planner.get_desired_state(DT)
            ic = 0.0
            conveyor_offset += 500.0 * DT
            if conveyor_offset > 900.0:
                done = True

        step += 1
        if step % LOG_SUBSAMPLE == 0:
            q_des_list.append(q_des.copy())
            ic_list.append(ic)

    return (
        np.array(q_des_list, dtype=np.float32),
        np.array(ic_list,    dtype=np.float32),
    )


# ── Corner extraction ─────────────────────────────────────────────────────────

def _ideal_corners_from_waypoints(waypoints: list) -> np.ndarray:
    """Déduplique les waypoints avec is_cutting=True → 4 coins idéaux (4, 2)."""
    seen = []
    for wp in waypoints:
        if wp[2]:
            pt = (round(wp[0], 9), round(wp[1], 9))
            if pt not in seen:
                seen.append(pt)
    assert len(seen) == 4, f"Attendu 4 coins, trouvé {len(seen)}"
    return np.array(seen, dtype=np.float32)


def _extract_real_corners(
    ideal_corners: np.ndarray,
    ee_des: np.ndarray,
    ee_real: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Pour chaque coin idéal, trouve le pas où FK(q_des) est le plus proche,
    puis lit FK(q_real) à ce pas. Applique l'inset vers le centre et le
    bruit de placement du convoyeur (reproduit piece_input.py).
    """
    real = np.zeros((4, 2), dtype=np.float32)
    for i, corner in enumerate(ideal_corners):
        t = int(np.argmin(np.linalg.norm(ee_des - corner, axis=1)))
        real[i] = ee_real[t]

    # Inset vers le centre
    center = real.mean(axis=0)
    for i in range(4):
        d = center - real[i]
        n = np.linalg.norm(d)
        if n > 1e-9:
            real[i] += (d / n) * DRILL_INSET

    # Bruit de placement convoyeur
    real += rng.normal(0.0, PLACEMENT_NOISE_SD, (4, 2)).astype(np.float32)
    return real


# ── World model loading ───────────────────────────────────────────────────────

def _load_model():
    device = (torch.device("mps")  if torch.backends.mps.is_available()  else
              torch.device("cuda") if torch.cuda.is_available()           else
              torch.device("cpu"))

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    H    = ckpt["hyperparams"]

    model = WorldModel(
        shape_embed_dim = H["shape_embed_dim"],
        h_dim           = H["h_dim"],
        obs_dim         = H["obs_dim"],
        dropout         = 0.0,
        gru_layers      = H["gru_layers"],
        pe_dim          = H.get("pe_dim", 64),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)

    normalizer = Normalizer.load(NORM_PATH)
    target_keys = H.get("target_keys", ["q_real"])
    if isinstance(target_keys, str):
        target_keys = [k.strip() for k in target_keys.split(",")]

    return model, normalizer, target_keys, device


# ── Main simulation ───────────────────────────────────────────────────────────

def simulate_cuts(waypoints: list, speeds: list) -> list:
    model, normalizer, target_keys, device = _load_model()
    predicts_error = target_keys == ["q_error"]

    ideal_corners = _ideal_corners_from_waypoints(waypoints)
    wp_arr = np.array(waypoints, dtype=np.float32)
    W = wp_arr.shape[0]
    rng = np.random.default_rng(42)

    results = []
    for speed in speeds:
        # ── 1. Génère q_des depuis le planificateur (sans physique) ──────────
        q_des, is_cutting = _generate_q_des(waypoints, speed)
        T = len(q_des)

        # ── 2. Inférence world model ─────────────────────────────────────────
        wps_t  = torch.zeros(1, W, 3, device=device)
        wps_t[0, :W] = torch.from_numpy(wp_arr).to(device)
        wp_len = torch.tensor([W], device=device)
        spd_t  = torch.tensor([[speed]], dtype=torch.float32, device=device)

        with torch.no_grad():
            pred_norm, quality = model.predict(wps_t, wp_len, spd_t, max_len=T)

        pred_np = normalizer.denormalize_tensor(pred_norm[0]).cpu().numpy()  # (T, 2)

        # ── 3. Reconstruit q_real ────────────────────────────────────────────
        if predicts_error:
            q_real = q_des + pred_np          # q_real = q_des + q_error
        else:
            q_real = pred_np                   # q_real directement

        # ── 4. FK + déviations ───────────────────────────────────────────────
        ee_real = _fk(q_real)
        ee_des  = _fk(q_des)
        deviation = np.linalg.norm(ee_real - ee_des, axis=1)  # (T,) en mètres

        # ── 5. Stats défauts (uniquement pendant la découpe) ─────────────────
        cut_mask = is_cutting > 0.5
        n_cut    = int(cut_mask.sum())

        if n_cut > 0:
            devs    = deviation[cut_mask]
            mean_mm = float(devs.mean()) * 1000.0
            p95_mm  = float(np.percentile(devs, 95)) * 1000.0
            max_mm  = float(devs.max())  * 1000.0
        else:
            mean_mm = 0.0
            p95_mm  = 0.0
            max_mm  = 0.0

        # ── 6. Coins réels pour le perçage ───────────────────────────────────
        real_corners = _extract_real_corners(ideal_corners, ee_des, ee_real, rng)

        results.append({
            "speed":             round(float(speed), 3),
            "mean_deviation_mm": round(mean_mm, 2),
            "p95_deviation_mm":  round(p95_mm, 2),
            "max_deviation_mm":  round(max_mm, 2),
            "n_cut_steps":       n_cut,
            "real_corners":      real_corners.tolist(),
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--waypoints", required=True)
    parser.add_argument("--speeds",    required=True)
    args = parser.parse_args()

    waypoints = json.loads(args.waypoints)
    speeds    = json.loads(args.speeds)

    print(json.dumps(simulate_cuts(waypoints, speeds)))


if __name__ == "__main__":
    main()
