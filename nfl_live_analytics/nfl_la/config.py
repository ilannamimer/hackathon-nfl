"""Constantes, chemins et palette de couleurs."""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def find_data_dir() -> Path | None:
    """Cherche le dossier `data/` du dépôt officiel (variable d'env NFL_DATA_DIR en priorité)."""
    candidates = [
        os.environ.get("NFL_DATA_DIR"),
        ROOT / "nfl-big-data-bowl-regional-event-data" / "data",
        ROOT / "data",
        ROOT.parent / "repo" / "data",
    ]
    for c in candidates:
        if c and (Path(c) / "plays.csv").exists():
            return Path(c)
    return None


FIELD_LEN, FIELD_WID = 120.0, 53.3   # yards (end zones incluses)
FPS = 10.0                           # fréquence du tracking (Hz)

# Palette « broadcast »
C_BG, C_PANEL = "#0B1220", "#121C30"
C_OFF, C_DEF, C_BALL = "#2F9BFF", "#FF4D5E", "#FFC933"
C_ACCENT, C_GOOD, C_WARN, C_TXT, C_MUTED = "#00E0B8", "#3DDC84", "#FFB020", "#EAF1FF", "#8CA0C3"
C_TURF, C_TURF2 = "#1C5A3A", "#1F6542"
