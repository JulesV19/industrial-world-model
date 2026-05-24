# piece.py
import random
import math
import json

_MAX_REACH = 2 * math.sqrt(2) - 0.05
_XMIN, _XMAX = 0.1, 1.9
_YMIN, _YMAX = 0.1, 1.9


def _rotate(pts, angle):
    c, s = math.cos(angle), math.sin(angle)
    return [(x*c - y*s, x*s + y*c) for x, y in pts]


def _clamp_to_workspace(pts, bx, by):
    """Rescale uniformément pour que tous les points restent dans le workspace.
    On utilise toujours le même facteur en x et y pour préserver la géométrie
    (cercles → cercles, pas ellipses).
    """
    worlds = [(bx + x, by + y) for x, y in pts]
    ox = max(0, max(wx - _XMAX for wx, _ in worlds),
                  max(_XMIN - wx for wx, _ in worlds))
    oy = max(0, max(wy - _YMAX for _, wy in worlds),
                  max(_YMIN - wy for _, wy in worlds))
    if ox > 0 or oy > 0:
        ext_x = max(abs(x) for x, y in pts) or 1e-9
        ext_y = max(abs(y) for x, y in pts) or 1e-9
        # Facteur de réduction isotrope : on prend le plus contraignant
        # pour garantir que la forme reste dans le workspace sans déformation
        fx = max(0.05, (ext_x - ox) / ext_x) if ox > 0 else 1.0
        fy = max(0.05, (ext_y - oy) / ext_y) if oy > 0 else 1.0
        f  = min(fx, fy)          # ← même facteur pour x ET y
        pts = [(x * f, y * f) for x, y in pts]
    return pts


def _square(s):
    h = s / 2
    return [(-h, -h), (h, -h), (h, h), (-h, h)]


def _circle(s, n_pts=48):
    """Cercle discrétisé en n_pts points équirépartis.
    48 points → erreur de corde < 0.07% du rayon, visuellement parfait.
    """
    r = s / 2
    return [(r * math.cos(i * 2*math.pi / n_pts),
             r * math.sin(i * 2*math.pi / n_pts)) for i in range(n_pts)]


def _triangle(s):
    r = s / math.sqrt(3)   # rayon du cercle circonscrit d'un triangle équilatéral de côté s
    pts = [(r * math.cos(math.pi/2 + i * 2*math.pi/3),
            r * math.sin(math.pi/2 + i * 2*math.pi/3)) for i in range(3)]
    angle = random.uniform(0, 2 * math.pi)
    return _rotate(pts, angle)


_SHAPES = [
    ("SQUARE",   _square),
    #("CIRCLE",   _circle),
    #("TRIANGLE", _triangle),
]
_SHAPE_NAMES = [n for n, _ in _SHAPES]
_SHAPE_FNS   = {n: fn for n, fn in _SHAPES}


def generate_shape_database(num_shapes=100):
    db = []

    for _ in range(num_shapes):
        base_x, base_y = 1.0, 1.0
        scale = random.uniform(1.2, 1.6)

        shape_type = random.choice(_SHAPE_NAMES)
        pts = _SHAPE_FNS[shape_type](scale)

        pts = _clamp_to_workspace(pts, base_x, base_y)

        _HOME = (0.1, 0.1)
        path = [
            (_HOME[0], _HOME[1], False),
            (base_x,   _YMIN,   False),
        ]

        for i, (lx, ly) in enumerate(pts):
            path.append((base_x + lx, base_y + ly, i > 0))

        lx0, ly0 = pts[0]
        path.append((base_x + lx0, base_y + ly0, True))

        path.append((base_x,   _YMIN,   False))
        path.append((_HOME[0], _HOME[1], False))

        db.append(path)

    return db


def load_shape_database(filename="pieces_database.json"):
    with open(filename, "r") as f:
        return json.load(f)
