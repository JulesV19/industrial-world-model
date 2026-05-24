"""
CLI runner — simulation de perçage via le world model.

Usage:
  python simulate_cli.py --corners '[[x,y],...]' --speeds '[0.5, 1.0]'

Les corners doivent provenir du résultat de simulate_cli.py de découpe
(champ "real_corners"), pas des coins géométriques idéaux.

Retourne JSON :
  [{"speed": 1.0, "n_defects": 1, "mean_error_mm": 11.2,
    "max_error_mm": 21.3, "per_corner_error_mm": [...]}, ...]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from world_model.model import DrillWorldModel

CKPT_PATH      = os.path.join(os.path.dirname(__file__), "world_model", "checkpoints", "best_model.pt")
DRILL_DEFECT_THR_M = 0.02  # 20 mm (même seuil que la physique)


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model():
    device = (torch.device("mps")  if torch.backends.mps.is_available()  else
              torch.device("cuda") if torch.cuda.is_available()           else
              torch.device("cpu"))

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    H    = ckpt["hyperparams"]

    model = DrillWorldModel(
        corner_embed_dim = H.get("corner_embed_dim", 64),
        embed_dim        = H.get("embed_dim",        256),
        h_dim            = H.get("h_dim",            512),
        pe_dim           = H.get("pe_dim",            64),
        gru_layers       = H.get("gru_layers",         2),
        n_attn_heads     = H.get("n_attn_heads",        4),
        dropout          = 0.0,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)

    # Normalizer pour les offsets (drill_hit - corner_target)
    norm_mean = torch.tensor(ckpt["norm_mean"], dtype=torch.float32, device=device)
    norm_std  = torch.tensor(ckpt["norm_std"],  dtype=torch.float32, device=device)

    return model, norm_mean, norm_std, device


# ── Main simulation ───────────────────────────────────────────────────────────

def simulate_drills(corners: list, speeds: list) -> list:
    model, norm_mean, norm_std, device = _load_model()

    corners_np  = np.array(corners, dtype=np.float32)        # (4, 2)
    corners_t   = torch.from_numpy(corners_np).unsqueeze(0).to(device)  # (1, 4, 2)

    results = []
    for speed in speeds:
        speed_t = torch.tensor([[speed]], dtype=torch.float32, device=device)

        with torch.no_grad():
            _, offsets_norm, defect_logits = model(corners_t, speed_t)

        # Dénormalise les offsets → positions réelles des trous
        offsets = offsets_norm[0] * norm_std + norm_mean     # (4, 2)
        offsets_np    = offsets.cpu().numpy()
        drill_hits_np = corners_np + offsets_np               # (4, 2)

        # Erreurs de positionnement
        errors_m  = np.linalg.norm(drill_hits_np - corners_np, axis=1)  # (4,) en mètres

        # Défauts via les logits
        defect_probs = torch.sigmoid(defect_logits[0]).cpu().numpy()     # (4,)
        n_defects    = int((defect_probs > 0.5).sum())

        results.append({
            "speed":               round(float(speed), 3),
            "mean_error_mm":       round(float(errors_m.mean()) * 1000, 2),
            "max_error_mm":        round(float(errors_m.max())  * 1000, 2),
            "per_corner_error_mm": [round(float(e) * 1000, 2) for e in errors_m],
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corners", required=True)
    parser.add_argument("--speeds",  required=True)
    args = parser.parse_args()

    corners = json.loads(args.corners)
    speeds  = json.loads(args.speeds)

    print(json.dumps(simulate_drills(corners, speeds)))


if __name__ == "__main__":
    main()
