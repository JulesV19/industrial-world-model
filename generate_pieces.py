import json
import argparse
from piece import generate_shape_database


def main():
    parser = argparse.ArgumentParser(description="Générateur de la base de données de pièces.")
    parser.add_argument("-n", "--num", type=int, default=100, help="Nombre de pièces à générer")
    args = parser.parse_args()
    
    num_pieces = args.num

    print(f"Génération de la base de données de {num_pieces} pièces...")
    # Génération des pièces
    db = generate_shape_database(num_pieces)
    
    # Sauvegarde dans un fichier JSON
    with open("pieces_database.json", "w") as f:
        json.dump(db, f, indent=4)
        
    print("Base de données sauvegardée avec succès dans 'pieces_database.json'.")

if __name__ == "__main__":
    main()
