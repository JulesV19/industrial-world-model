import json
import random
import argparse
from piece import generate_shape_database


def main():
    parser = argparse.ArgumentParser(description="Générateur de la base de données de pièces.")
    parser.add_argument("-n", "--num",       type=int,   default=100,
                        help="Nombre de pièces à générer")
    parser.add_argument("--seed",            type=int,   default=42,
                        help="Graine aléatoire (reproductibilité)")
    parser.add_argument("--n-runs",          type=int,   default=5,
                        help="Nombre de runs par pièce (chaque run tire une vitesse aléatoire)")
    parser.add_argument("--speed-min",       type=float, default=1.0,
                        help="Durée minimale par segment (s) — cadence maximale")
    parser.add_argument("--speed-max",       type=float, default=3.0,
                        help="Durée maximale par segment (s) — cadence minimale")
    args = parser.parse_args()

    assert args.speed_min < args.speed_max, "speed-min doit être < speed-max"

    random.seed(args.seed)

    print(f"Génération de {args.num} pièces × {args.n_runs} runs, "
          f"vitesse ∈ U[{args.speed_min}, {args.speed_max}] s/seg...")
    pieces = generate_shape_database(args.num)

    db = {
        "pieces":    pieces,
        "n_runs":    args.n_runs,
        "speed_min": args.speed_min,
        "speed_max": args.speed_max,
    }

    with open("pieces_database.json", "w") as f:
        json.dump(db, f, indent=4)

    total = args.num * args.n_runs
    print(f"Base sauvegardée : {args.num} pièces × {args.n_runs} runs = {total} épisodes à enregistrer.")


if __name__ == "__main__":
    main()
