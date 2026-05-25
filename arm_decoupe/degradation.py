from config import FRICTION_DEGRAD_ALPHA, FRICTION_DEGRAD_HALFLIFE, TEMP_NOISE_GAMMA, CADENCE_REF


def friction_multiplier(piece_count: int) -> float:
    """Saturation Hill : 1.0 à neuf → 1+alpha à saturation, moitié à piece_count=halflife."""
    return 1.0 + FRICTION_DEGRAD_ALPHA * piece_count / (piece_count + FRICTION_DEGRAD_HALFLIFE)


def noise_multiplier(cadence: float) -> float:
    """Linéaire avec la cadence (pièces/heure) : 1.0 à l'arrêt → 1+gamma à cadence_ref."""
    return 1.0 + TEMP_NOISE_GAMMA * (cadence / CADENCE_REF)
