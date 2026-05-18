# piece.py
import random
import math
import json

def load_shape_database(filename="pieces_database.json"):
    with open(filename, "r") as f:
        return json.load(f)

def generate_shape_database(num_shapes=100):
    db = []
    safe_z = 0.6 # Position de repli (hauteur)

    for _ in range(num_shapes):
        # Position aléatoire du centre de la pièce dans l'espace de travail
        base_x = random.uniform(0.85, 1.15)
        base_y = random.uniform(-0.3, 0.2)
        
        # Rotation aléatoire de la pièce
        rotation = random.uniform(0, 2 * math.pi)

        shape_type = random.choice(["L_SHAPE", "RECTANGLE", "TRIANGLE", "HEXAGON", "STAR", "CIRCLE", "RANDOM_POLYGON"])
        scale = random.uniform(0.3, 0.7) 
        
        path = [(0.4, safe_z, False), (base_x, safe_z, False)] # Approche
        
        def add_points(pts_local):
            for i, (lx, ly) in enumerate(pts_local):
                # Application de la rotation et translation
                rx = lx * math.cos(rotation) - ly * math.sin(rotation)
                ry = lx * math.sin(rotation) + ly * math.cos(rotation)
                path.append((base_x + rx, base_y + ry, i > 0)) # i=0 -> Descente sans couper
            
            # Fermeture de la forme (retour au point initial avec coupe)
            lx, ly = pts_local[0]
            rx = lx * math.cos(rotation) - ly * math.sin(rotation)
            ry = lx * math.sin(rotation) + ly * math.cos(rotation)
            path.append((base_x + rx, base_y + ry, True))

        pts = []
        if shape_type == "RECTANGLE":
            w, h = scale, scale * random.uniform(0.4, 1.8)
            pts = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
            
        elif shape_type == "L_SHAPE":
            th = scale * random.uniform(0.2, 0.4)
            pts = [(-scale/2, -scale/2), (scale/2, -scale/2), (scale/2, -scale/2 + th),
                   (-scale/2 + th, -scale/2 + th), (-scale/2 + th, scale/2), (-scale/2, scale/2)]
            
        elif shape_type == "TRIANGLE":
            pts = [(-scale/2, -scale/2), (scale/2, -scale/2), (random.uniform(-scale/4, scale/4), scale/2)]
            
        elif shape_type == "HEXAGON":
            for i in range(6):
                angle = i * (math.pi / 3)
                pts.append((scale/2 * math.cos(angle), scale/2 * math.sin(angle)))

        elif shape_type == "STAR":
            points = random.choice([4, 5, 6, 7])
            outer_r = scale / 2
            inner_r = outer_r * random.uniform(0.3, 0.6)
            for i in range(points * 2):
                r = outer_r if i % 2 == 0 else inner_r
                angle = i * (math.pi / points)
                pts.append((r * math.cos(angle), r * math.sin(angle)))

        elif shape_type == "CIRCLE":
            resolution = random.randint(12, 24)
            r = scale / 2
            for i in range(resolution):
                angle = i * (2 * math.pi / resolution)
                pts.append((r * math.cos(angle), r * math.sin(angle)))

        elif shape_type == "RANDOM_POLYGON":
            points = random.randint(4, 7)
            angles = sorted([random.uniform(0, 2 * math.pi) for _ in range(points)])
            for angle in angles:
                r = (scale/2) * random.uniform(0.5, 1.0)
                pts.append((r * math.cos(angle), r * math.sin(angle)))

        # Ajout d'un bruit sur les sommets pour déformer légèrement (imperfections de conception)
        noise_level = random.uniform(0.0, 0.05) if shape_type != "CIRCLE" else 0.0
        noisy_pts = [(x + random.uniform(-noise_level, noise_level), y + random.uniform(-noise_level, noise_level)) for x, y in pts]

        add_points(noisy_pts)
                
        path.append((base_x, safe_z, False)) # Dégagement vertical
        path.append((0.4, safe_z, False))    # Retour point de repos
        db.append(path)
        
    return db
