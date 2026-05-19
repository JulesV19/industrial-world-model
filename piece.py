# piece.py
import random
import math
import json

# Portée max du bras (L1+L2 = 2√2 ≈ 2.83)
_MAX_REACH = 2 * math.sqrt(2) - 0.05   # marge de sécurité

# Carré de travail : les pièces doivent rester dans [0.1, 1.9]²
_XMIN, _XMAX = 0.1, 1.9
_YMIN, _YMAX = 0.1, 1.9


# ── utilitaires géométriques ──────────────────────────────────────────────────

def _rotate(pts, angle):
    c, s = math.cos(angle), math.sin(angle)
    return [(x*c - y*s, x*s + y*c) for x, y in pts]


def _clamp_to_workspace(pts, bx, by):
    """Réduit les coords locales si un sommet sort du carré [_XMIN,_XMAX]x[_YMIN,_YMAX]."""
    worlds = [(bx + x, by + y) for x, y in pts]
    # dépassement dans chaque direction
    ox = max(0, max(wx - _XMAX for wx, _ in worlds),
                  max(_XMIN - wx for wx, _ in worlds))
    oy = max(0, max(wy - _YMAX for _, wy in worlds),
                  max(_YMIN - wy for _, wy in worlds))
    if ox > 0 or oy > 0:
        # facteur de réduction conservatif
        ext_x = max(abs(x) for x, y in pts) or 1e-9
        ext_y = max(abs(y) for x, y in pts) or 1e-9
        fx = max(0.05, (ext_x - ox) / ext_x) if ox > 0 else 1.0
        fy = max(0.05, (ext_y - oy) / ext_y) if oy > 0 else 1.0
        f  = min(fx, fy)
        pts = [(x * f, y * f) for x, y in pts]
    return pts


def _center(pts):
    """Centre les points locaux autour de (0,0)."""
    mx = sum(x for x, y in pts) / len(pts)
    my = sum(y for x, y in pts) / len(pts)
    return [(x - mx, y - my) for x, y in pts]


# ── générateurs de formes (coords locales, centrées en 0) ────────────────────

def _rectangle(s):
    w = s
    h = s * random.uniform(0.25, 2.8)
    hw, hh = w/2, h/2
    return [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]


def _triangle(s):
    style = random.choice(["iso", "right", "scalene", "obtuse"])
    if style == "iso":
        apex_x = random.uniform(-s*0.25, s*0.25)
        return [(-s/2, -s/2), (s/2, -s/2), (apex_x, s/2)]
    elif style == "right":
        ax = s * random.uniform(0.4, 1.0)
        ay = s * random.uniform(0.4, 1.0)
        return [(0, 0), (ax, 0), (0, ay)]
    elif style == "scalene":
        pts = [(random.uniform(-s/2, s/2), random.uniform(-s/2, s/2)) for _ in range(3)]
        return _center(pts)
    else:  # obtuse
        return [(-s*0.65, -s/4), (s*0.65, -s/4),
                (s*random.uniform(-0.6, 0.0), s/4)]


def _trapezoid(s):
    w_bot = s * random.uniform(0.6, 1.3)
    w_top = s * random.uniform(0.15, 0.85)
    h = s * random.uniform(0.35, 1.1)
    offset = s * random.uniform(-0.35, 0.35)
    return [(-w_bot/2, -h/2), (w_bot/2, -h/2),
            (offset + w_top/2, h/2), (offset - w_top/2, h/2)]


def _parallelogram(s):
    w = s * random.uniform(0.7, 1.4)
    h = s * random.uniform(0.3, 1.0)
    shear = s * random.uniform(0.15, 0.5)
    return [(-w/2, -h/2), (w/2 - shear, -h/2),
            (w/2, h/2),   (-w/2 + shear, h/2)]


def _regular_polygon(s):
    n = random.randint(5, 10)
    r = s / 2
    off = random.uniform(0, math.pi / n)
    return [(r * math.cos(i * 2*math.pi/n + off),
             r * math.sin(i * 2*math.pi/n + off)) for i in range(n)]


def _ellipse(s):
    rx = s / 2
    ry = rx * random.uniform(0.25, 2.2)
    n = random.randint(12, 26)
    return [(rx * math.cos(i * 2*math.pi/n),
             ry * math.sin(i * 2*math.pi/n)) for i in range(n)]


def _sector(s):
    """Secteur angulaire (camembert)."""
    r = s / 2
    span = random.uniform(math.pi/5, 5*math.pi/3)
    start = random.uniform(0, 2*math.pi)
    n = random.randint(8, 18)
    pts = [(0.0, 0.0)]
    for i in range(n + 1):
        a = start + i * span / n
        pts.append((r * math.cos(a), r * math.sin(a)))
    return _center(pts)


def _l_shape(s):
    th = s * random.uniform(0.15, 0.42)
    fx, fy = random.choice([1, -1]), random.choice([1, -1])
    pts = [(-s/2, -s/2), (s/2, -s/2), (s/2, -s/2 + th),
           (-s/2 + th, -s/2 + th), (-s/2 + th, s/2), (-s/2, s/2)]
    return [(x*fx, y*fy) for x, y in pts]


def _t_shape(s):
    bar_h  = s * random.uniform(0.18, 0.38)
    stem_w = s * random.uniform(0.18, 0.42)
    stem_h = s * random.uniform(0.35, 0.70)
    bar_w  = s
    off    = s * random.uniform(-0.18, 0.18)   # tige décalée
    top = s/2; bb = s/2 - bar_h; sb = bb - stem_h
    pts = [
        (-bar_w/2, top), (bar_w/2, top),
        (bar_w/2, bb),
        (off + stem_w/2, bb), (off + stem_w/2, sb),
        (off - stem_w/2, sb), (off - stem_w/2, bb),
        (-bar_w/2, bb),
    ]
    return _center(pts)


def _u_shape(s):
    wall = s * random.uniform(0.12, 0.30)
    depth = s * random.uniform(0.40, 0.72)
    inner_w = s * random.uniform(0.22, 0.55)
    outer_w = inner_w + 2 * wall
    pts = [
        (-outer_w/2, -s/2), (outer_w/2, -s/2),
        (outer_w/2, -s/2 + depth),
        (outer_w/2 - wall, -s/2 + depth),
        (outer_w/2 - wall, -s/2 + wall),
        (-(outer_w/2 - wall), -s/2 + wall),
        (-(outer_w/2 - wall), -s/2 + depth),
        (-outer_w/2, -s/2 + depth),
    ]
    return _center(pts)


def _c_shape(s):
    pts = _u_shape(s)
    return [(-y, x) for x, y in pts]   # rotation 90°


def _cross(s):
    arm_w = s * random.uniform(0.18, 0.48)
    h, hw = s/2, arm_w/2
    return [
        (-hw, -h), (hw, -h),
        (hw, -hw), (h, -hw),
        (h,  hw),  (hw, hw),
        (hw,  h),  (-hw,  h),
        (-hw, hw), (-h,  hw),
        (-h, -hw), (-hw, -hw),
    ]


def _arrow(s):
    head_w = s * random.uniform(0.5, 0.95)
    head_h = s * random.uniform(0.28, 0.55)
    shaft_w = s * random.uniform(0.14, 0.38)
    return [
        (0, s/2),
        (head_w/2, s/2 - head_h),
        (shaft_w/2, s/2 - head_h),
        (shaft_w/2, -s/2),
        (-shaft_w/2, -s/2),
        (-shaft_w/2, s/2 - head_h),
        (-head_w/2, s/2 - head_h),
    ]


def _star_regular(s):
    n = random.randint(4, 8)
    R = s / 2
    r = R * random.uniform(0.22, 0.68)
    pts = []
    for i in range(n * 2):
        radius = R if i % 2 == 0 else r
        a = i * math.pi / n
        pts.append((radius * math.cos(a), radius * math.sin(a)))
    return pts


def _star_irregular(s):
    """Étoile avec rayons extérieurs variables par pointe."""
    n = random.randint(4, 9)
    R = s / 2
    r_inner = R * random.uniform(0.18, 0.50)
    pts = []
    for i in range(n):
        R_i = R * random.uniform(0.55, 1.0)
        a_out = i * 2*math.pi / n + random.uniform(-0.12, 0.12)
        a_in  = a_out + math.pi / n
        pts.append((R_i * math.cos(a_out), R_i * math.sin(a_out)))
        ri = r_inner * random.uniform(0.65, 1.35)
        pts.append((ri * math.cos(a_in), ri * math.sin(a_in)))
    return pts


def _e_shape(s):
    spine_w = s * random.uniform(0.12, 0.24)
    bar_h   = s * random.uniform(0.10, 0.20)
    width   = s * random.uniform(0.55, 0.95)
    mid_w   = width * random.uniform(0.45, 0.90)
    gap     = (s - 3*bar_h) / 2
    x0 = -s/2
    pts = [
        (x0, -s/2),
        (x0 + width, -s/2),
        (x0 + width, -s/2 + bar_h),
        (x0 + spine_w, -s/2 + bar_h),
        (x0 + spine_w, -s/2 + bar_h + gap),
        (x0 + mid_w, -s/2 + bar_h + gap),
        (x0 + mid_w, -s/2 + 2*bar_h + gap),
        (x0 + spine_w, -s/2 + 2*bar_h + gap),
        (x0 + spine_w, -s/2 + 2*bar_h + 2*gap),
        (x0 + width, -s/2 + 2*bar_h + 2*gap),
        (x0 + width, s/2),
        (x0, s/2),
    ]
    return _center(pts)


def _comb(s):
    n_teeth = random.randint(2, 5)
    tooth_h = s * random.uniform(0.28, 0.60)
    tooth_w_frac = random.uniform(0.35, 0.65)   # fraction de l'espace par dent
    total_slots = 2 * n_teeth + 1
    slot_w = s / total_slots
    tooth_w = slot_w * tooth_w_frac
    gap_w   = slot_w * (2 - tooth_w_frac)       # gap entre dents
    base_h  = s - tooth_h

    pts = [(-s/2, -s/2), (-s/2, -s/2 + base_h)]
    for i in range(n_teeth):
        tx = -s/2 + gap_w * (i + 0.5) + (tooth_w + gap_w) * i
        pts.extend([
            (tx, -s/2 + base_h),
            (tx, s/2),
            (tx + tooth_w, s/2),
            (tx + tooth_w, -s/2 + base_h),
        ])
    pts.extend([(s/2, -s/2 + base_h), (s/2, -s/2)])
    return _center(pts)


def _zigzag(s):
    n = random.randint(3, 7)
    amp   = s * random.uniform(0.18, 0.45)
    thick = s * random.uniform(0.08, 0.22)
    step  = s / n
    pts   = []
    for i in range(n + 1):
        y = -s/2 + i * step
        x = amp/2 if i % 2 == 0 else -amp/2
        pts.append((x, y))
    for i in range(n, -1, -1):
        y = -s/2 + i * step
        x = amp/2 if i % 2 == 0 else -amp/2
        pts.append((x - thick, y))
    return _center(pts)


def _random_polygon(s):
    n = random.randint(5, 14)
    angles = sorted(random.uniform(0, 2*math.pi) for _ in range(n))
    return [(s/2 * random.uniform(0.25, 1.0) * math.cos(a),
             s/2 * random.uniform(0.25, 1.0) * math.sin(a)) for a in angles]


def _blob(s):
    """Polygone lisse aux contours organiques."""
    n = random.randint(9, 18)
    angles = [i * 2*math.pi / n for i in range(n)]
    radii  = [s/2 * random.uniform(0.35, 1.0) for _ in range(n)]
    for _ in range(4):   # lissage
        radii = [(radii[i-1] + 2*radii[i] + radii[(i+1) % n]) / 4 for i in range(n)]
    return [(radii[i] * math.cos(angles[i]),
             radii[i] * math.sin(angles[i])) for i in range(n)]


# ── table des formes ──────────────────────────────────────────────────────────

_SHAPES = [
    ("RECTANGLE",       _rectangle),
    ("TRIANGLE",        _triangle),
    ("TRAPEZOID",       _trapezoid),
    ("PARALLELOGRAM",   _parallelogram),
    ("REGULAR_POLYGON", _regular_polygon),
    ("ELLIPSE",         _ellipse),
    ("SECTOR",          _sector),
    ("L_SHAPE",         _l_shape),
    ("T_SHAPE",         _t_shape),
    ("U_SHAPE",         _u_shape),
    ("C_SHAPE",         _c_shape),
    ("CROSS",           _cross),
    ("ARROW",           _arrow),
    ("STAR",            _star_regular),
    ("IRREGULAR_STAR",  _star_irregular),
    ("E_SHAPE",         _e_shape),
    ("COMB",            _comb),
    ("ZIGZAG",          _zigzag),
    ("RANDOM_POLYGON",  _random_polygon),
    ("BLOB",            _blob),
]
_SHAPE_NAMES = [n for n, _ in _SHAPES]
_SHAPE_FNS   = {n: fn for n, fn in _SHAPES}


# ── générateur principal ──────────────────────────────────────────────────────

def generate_shape_database(num_shapes=100):
    db = []

    for _ in range(num_shapes):
        # ── position du centre : toujours au centre du workspace ──────────────
        base_x, base_y = 1.0, 1.0   # centre de [0,2]²

        # ── scale de base (pièces larges ~50cm–1m, clampées dans [0.1,1.9]²) ──────
        scale = random.uniform(1.2, 1.6)

        # ── choix et génération de la forme ──────────────────────────────────
        shape_type = random.choice(_SHAPE_NAMES)
        pts = _SHAPE_FNS[shape_type](scale)

        # ── déformations additionnelles ───────────────────────────────────────
        # Echelle non-uniforme (anisotropie)
        if random.random() < 0.45:
            sx = random.uniform(0.55, 1.60)
            sy = random.uniform(0.55, 1.60)
            pts = [(x * sx, y * sy) for x, y in pts]

        # Cisaillement
        if random.random() < 0.25:
            shear = random.uniform(-0.15, 0.15)
            pts = [(x + shear * y, y) for x, y in pts]

        # Bruit sommet (imperfections de conception)
        noise = random.uniform(0.0, 0.08) * scale
        pts = [(x + random.uniform(-noise, noise),
                y + random.uniform(-noise, noise)) for x, y in pts]

        # ── rotation aléatoire ────────────────────────────────────────────────
        pts = _rotate(pts, random.uniform(0, 2*math.pi))

        # ── contrainte workspace (carré [0.1, 1.9]²) ───────────────────────────
        pts = _clamp_to_workspace(pts, base_x, base_y)


        # Home : position hors singularité — (0.1, 0.1) est le coin bas-gauche du workspace
        # (0,0) exact est une singularité IK : l'IK n'y est pas continue, causant
        # une discontinuité de q1 au premier mouvement.
        _HOME = (0.1, 0.1)
        path = [
            (_HOME[0], _HOME[1], False),  # home (bras replié, non-singulier)
            (base_x,   _YMIN,   False),   # approche colonne (bas du carré)
        ]

        for i, (lx, ly) in enumerate(pts):
            path.append((base_x + lx, base_y + ly, i > 0))

        # Fermeture de la forme
        lx0, ly0 = pts[0]
        path.append((base_x + lx0, base_y + ly0, True))

        path.append((base_x,   _YMIN,   False))  # recul vers le bas
        path.append((_HOME[0], _HOME[1], False))  # retour home

        db.append(path)

    return db


# ── chargement ────────────────────────────────────────────────────────────────

def load_shape_database(filename="pieces_database.json"):
    with open(filename, "r") as f:
        return json.load(f)
