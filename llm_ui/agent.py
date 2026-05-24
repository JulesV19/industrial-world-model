"""
Agent LangChain + Mistral avec tool use pour l'optimisation de production.
"""
import json
import os
import subprocess
import uuid
from typing import Generator

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECOUPE_DIR = os.path.join(BASE_DIR, "arm_decoupe")
PERCAGE_DIR = os.path.join(BASE_DIR, "arm_percage")
PYTHON      = os.path.join(BASE_DIR, ".venv", "bin", "python3")

# ── Session state (in-memory) ────────────────────────────────────────────────
# { session_id: [LangChain messages] }
_sessions: dict[str, list] = {}


def get_or_create_session(session_id: str | None) -> tuple[str, list]:
    if not session_id or session_id not in _sessions:
        session_id = str(uuid.uuid4())
        _sessions[session_id] = []
    return session_id, _sessions[session_id]


def reset_session(session_id: str) -> None:
    _sessions.pop(session_id, None)


# ── Simulation helpers ───────────────────────────────────────────────────────

def _run_subprocess(cwd: str, script: str, timeout: int = 180, **kwargs) -> str:
    cmd = [PYTHON, script]
    for k, v in kwargs.items():
        cmd += [f"--{k.replace('_', '-')}", v]
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        return json.dumps({"error": result.stderr[:800]})
    return result.stdout.strip()


def _generate_square(side_meters: float) -> dict:
    _HOME  = [0.1, 0.1]
    bx, by = 1.0, 1.0
    _XMIN, _XMAX = 0.1, 1.9
    _YMIN, _YMAX = 0.1, 1.9
    h = side_meters / 2

    corners_local = [(-h, -h), (h, -h), (h, h), (-h, h)]

    worlds    = [(bx + x, by + y) for x, y in corners_local]
    exceed_x  = max((max(wx - _XMAX, _XMIN - wx, 0) for wx, _ in worlds), default=0)
    exceed_y  = max((max(wy - _YMAX, _YMIN - wy, 0) for _, wy in worlds), default=0)
    actual_side = side_meters

    if exceed_x > 0 or exceed_y > 0:
        fx = max(0.05, (h - exceed_x) / h) if exceed_x > 0 else 1.0
        fy = max(0.05, (h - exceed_y) / h) if exceed_y > 0 else 1.0
        f  = min(fx, fy)
        corners_local = [(x * f, y * f) for x, y in corners_local]
        actual_side   = side_meters * f

    path = [[_HOME[0], _HOME[1], False], [bx, _YMIN, False]]
    for i, (lx, ly) in enumerate(corners_local):
        path.append([round(bx + lx, 6), round(by + ly, 6), i > 0])
    lx0, ly0 = corners_local[0]
    path.append([round(bx + lx0, 6), round(by + ly0, 6), True])
    path.append([bx, _YMIN, False])
    path.append([_HOME[0], _HOME[1], False])

    drill_corners = [[round(bx + lx, 6), round(by + ly, 6)] for lx, ly in corners_local]

    note = f"Carré de {actual_side * 100:.1f} cm de côté, centré en ({bx}, {by})"
    if abs(actual_side - side_meters) > 0.001:
        note += (f" (réduit depuis {side_meters * 100:.1f} cm pour rester "
                 f"dans l'espace de travail)")

    return {
        "waypoints":        path,
        "drill_corners":    drill_corners,
        "requested_side_m": side_meters,
        "actual_side_m":    round(actual_side, 4),
        "note":             note,
    }


# ── Tool schemas (Pydantic) ──────────────────────────────────────────────────

class GenerateSquareInput(BaseModel):
    side_meters: float = Field(
        description="Longueur du côté en mètres (ex: 0.5 → 50 cm). Plage : 0.1–1.6 m."
    )


class SimulateCuttingInput(BaseModel):
    waypoints: list = Field(
        description="Les waypoints produits par generate_square."
    )
    speeds: list[float] = Field(
        description=(
            "Durées par segment à tester (secondes). "
            "Valeur faible → rapide mais défauts. Plage 0.4–4.0 s. Max 8 valeurs."
        )
    )


class SimulateDrillingInput(BaseModel):
    corners: list = Field(
        description=(
            "Les 4 coins réels [[x,y], ...] extraits du résultat de simulate_cutting "
            "(champ 'real_corners' d'une vitesse de découpe choisie). "
            "NE PAS utiliser les coins idéaux de generate_square."
        )
    )
    speeds: list[float] = Field(
        description="Durées par segment à tester (secondes). Max 8 valeurs."
    )


# ── Tool functions ───────────────────────────────────────────────────────────

def _tool_generate_square(side_meters: float) -> str:
    return json.dumps(_generate_square(side_meters))


def _tool_simulate_cutting(waypoints: list, speeds: list) -> str:
    return _run_subprocess(
        DECOUPE_DIR, "simulate_cli.py",
        waypoints=json.dumps(waypoints),
        speeds=json.dumps(speeds),
    )


def _tool_simulate_drilling(corners: list, speeds: list) -> str:
    return _run_subprocess(
        PERCAGE_DIR, "simulate_cli.py",
        corners=json.dumps(corners),
        speeds=json.dumps(speeds),
    )


TOOLS = [
    StructuredTool.from_function(
        func=_tool_generate_square,
        name="generate_square",
        description=(
            "Génère les waypoints de découpe et les coins de perçage pour un carré. "
            "Appelle cet outil en premier pour créer la pièce avant toute simulation."
        ),
        args_schema=GenerateSquareInput,
    ),
    StructuredTool.from_function(
        func=_tool_simulate_cutting,
        name="simulate_cutting",
        description=(
            "Simule la découpe via le world model pour plusieurs vitesses. "
            "Retourne pour chaque vitesse : déviation moyenne, p95 et max (en mm), "
            "ET les coins réels (real_corners) tels que le bras de découpe les a "
            "réellement atteints. Ces real_corners doivent être passés à simulate_drilling."
        ),
        args_schema=SimulateCuttingInput,
    ),
    StructuredTool.from_function(
        func=_tool_simulate_drilling,
        name="simulate_drilling",
        description=(
            "Simule le perçage via le world model pour plusieurs vitesses. "
            "Utilise les real_corners issus de simulate_cutting (PAS les coins idéaux). "
            "Retourne pour chaque vitesse : erreur moyenne, max et par coin (en mm)."
        ),
        args_schema=SimulateDrillingInput,
    ),
]

TOOL_MAP = {t.name: t for t in TOOLS}

SYSTEM_PROMPT = """Tu es un expert en optimisation de production pour une chaîne de fabrication robotisée 2-DOF.

## La chaîne de production

Elle comporte deux étapes séquentielles pour fabriquer des carrés métalliques :

**1. Découpe** — un bras 2-DOF découpe le contour du carré au laser.
- Qualité mesurée par la déviation cartésienne (mm) entre trajectoire réelle et désirée pendant la coupe
- Indicateurs : déviation moyenne, p95 (95e percentile) et max sur l'ensemble des pas de découpe

**2. Perçage** — un second bras perce les 4 coins de la pièce découpée.
- Qualité mesurée par l'erreur de positionnement (mm) par coin
- Indicateurs : erreur moyenne, max, et détail par coin

## Le paramètre clé : duration_per_segment (secondes)

Temps alloué à chaque segment de trajectoire :
- Valeur faible (0.4–1.0 s) → production rapide ⚡, mais inertie et vibrations → défauts ❌
- Valeur élevée (2.5–4.0 s) → production lente 🐢, contrôleur stabilisé → qualité ✅

Les deux bras ont des dynamiques différentes — la vitesse optimale pour la découpe peut différer de celle pour le perçage.

## Ta mission

1. Comprendre la pièce demandée (côté en mètres ou centimètres)
2. Comprendre le critère qualité de l'utilisateur (ex : "déviation découpe < 15 mm en moyenne", "erreur de perçage < 10 mm")
3. Explorer intelligemment la plage de vitesses :
   - Balayage grossier : 6–8 points uniformes sur 0.4–4.0 s
   - Affiner dans la zone de transition si nécessaire
4. Présenter les résultats sous forme de tableau clair
5. Recommander la vitesse optimale avec justification et compromis chiffrés

## Pipeline — important

La découpe et le perçage sont simulés par deux world models distincts. Les coins
passés au perçage **ne sont pas les coins géométriques idéaux** : ce sont les positions
réelles atteintes par le bras de découpe (avec ses erreurs), récupérées dans le champ
`real_corners` du résultat de `simulate_cutting`.

## Workflow recommandé

```
1. generate_square(side_meters)
   → waypoints

2. simulate_cutting(waypoints, [0.4, 0.7, 1.0, 1.5, 2.0, 2.8, 3.5])
   → pour chaque vitesse : mean_deviation_mm, p95_deviation_mm, max_deviation_mm, real_corners

3. Choisir une vitesse de découpe de référence (ou tester plusieurs)
   Récupérer ses real_corners

4. simulate_drilling(real_corners, [0.4, 0.7, 1.0, 1.5, 2.0, 2.8, 3.5])
   → pour chaque vitesse : n_defects, mean_error_mm, per_corner_error_mm

5. Analyser — affiner si la zone de transition est floue
6. Recommander avec tableau récapitulatif et compromis chiffrés
```

Exprime-toi en français. Sois précis sur les chiffres. Explique les compromis clairement."""


# ── Agent loop ───────────────────────────────────────────────────────────────

def run_agent(user_message: str, session_id: str | None) -> Generator[str, None, None]:
    """
    Yields SSE-formatted strings.
    Event types: session | text | tool_start | tool_result | error | done
    """
    mistral_key = os.environ.get("MISTRAL_API_KEY", "")
    if not mistral_key:
        msg = "MISTRAL_API_KEY non définie dans l'environnement."
        yield f"data: {json.dumps({'type': 'error', 'content': msg})}\n\n"
        return

    sid, history = get_or_create_session(session_id)
    yield f"data: {json.dumps({'type': 'session', 'session_id': sid})}\n\n"

    llm = ChatMistralAI(
        model="mistral-large-latest",
        api_key=mistral_key,
        temperature=0.1,
    )
    llm_with_tools = llm.bind_tools(TOOLS)

    history.append(HumanMessage(content=user_message))
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + history

    while True:
        response = llm_with_tools.invoke(messages)
        history.append(response)
        messages.append(response)

        # Stream text content
        if response.content:
            yield f"data: {json.dumps({'type': 'text', 'content': response.content})}\n\n"

        # No tool calls → done
        if not response.tool_calls:
            break

        # Execute tool calls
        for tc in response.tool_calls:
            tool_name  = tc["name"]
            tool_input = tc["args"]
            tool_id    = tc["id"]

            yield f"data: {json.dumps({'type': 'tool_start', 'tool': tool_name, 'input': tool_input})}\n\n"

            tool_fn = TOOL_MAP.get(tool_name)
            if tool_fn is None:
                raw = json.dumps({"error": f"Outil inconnu : {tool_name}"})
            else:
                try:
                    raw = tool_fn.invoke(tool_input)
                except Exception as exc:
                    raw = json.dumps({"error": str(exc)})

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = raw

            tool_msg = ToolMessage(content=raw, tool_call_id=tool_id)
            history.append(tool_msg)
            messages.append(tool_msg)

            yield f"data: {json.dumps({'type': 'tool_result', 'tool': tool_name, 'result': parsed})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"
