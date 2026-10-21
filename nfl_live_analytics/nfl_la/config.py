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

# Palette « NFL.com » (thème clair)
C_NAVY, C_RED = "#013369", "#D50A0A"
C_BG, C_PANEL = "#FFFFFF", "#F2F4F7"
C_OFF, C_DEF, C_BALL = "#013369", "#D50A0A", "#FFC933"
C_ACCENT, C_GOOD, C_WARN, C_TXT, C_MUTED = "#D50A0A", "#1E9E5A", "#F2A900", "#111827", "#5B6770"
C_TRAIL = "#FFD60A"
C_TURF, C_TURF2 = "#2E7D4F", "#33875A"
FONT = "Barlow, 'Barlow Condensed', Arial, sans-serif"