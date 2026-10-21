"""Chargement robuste des fichiers CSV du dépôt + normalisation des coordonnées.

Convention : après `normalize_tracking`, l'attaque va TOUJOURS vers la droite (x croissant)
et 0° = vers +y (convention NFL), ce qui rend toutes les actions comparables.
"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from .config import FIELD_WID

log = logging.getLogger("nfl_la")


class DataError(RuntimeError):
    """Erreur lisible pour l'utilisateur (fichier manquant, colonne absente…)."""


REQUIRED = {
    "games.csv": ["gameId", "week", "homeTeamAbbr", "visitorTeamAbbr"],
    "plays.csv": ["gameId", "playId", "down", "yardsToGo", "possessionTeam", "defensiveTeam",
                  "passResult", "prePenaltyPlayResult", "playResult", "quarter", "gameClock",
                  "preSnapHomeScore", "preSnapVisitorScore", "playDescription"],
    "players.csv": ["nflId", "displayName", "officialPosition"],
    "pffScoutingData.csv": ["gameId", "playId", "nflId", "pff_role", "pff_positionLinedUp"],
}
TRACK_COLS = ["gameId", "playId", "nflId", "frameId", "team", "playDirection", "x", "y", "s", "a", "dis",
              "o", "dir", "event", "jerseyNumber"]


def _read(data_dir: Path, name: str) -> pd.DataFrame:
    path = Path(data_dir) / name
    if not path.exists():
        raise DataError(f"Fichier introuvable : {path}. Clonez le dépôt officiel ou définissez NFL_DATA_DIR.")
    try:
        df = pd.read_csv(path)
    except Exception as e:  # CSV corrompu, encodage…
        raise DataError(f"Lecture impossible de {name} : {e}") from e
    missing = [c for c in REQUIRED.get(name, []) if c not in df.columns]
    if missing:
        raise DataError(f"Colonnes absentes dans {name} : {missing}")
    return df


def load_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Charge games / plays / players / pff (tables légères)."""
    t = {k.replace(".csv", ""): _read(data_dir, k) for k in REQUIRED}
    t["games"]["label"] = ("S" + t["games"].week.astype(str)).str.cat(
        t["games"].visitorTeamAbbr + " @ " + t["games"].homeTeamAbbr, sep=" · ")
    return t


def normalize_tracking(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne le terrain pour que l'attaque aille toujours vers la droite."""
    df = df.copy()
    left = (df["playDirection"] == "left").to_numpy()
    df["x"] = np.where(left, 120.0 - df["x"], df["x"])
    df["y"] = np.where(left, FIELD_WID - df["y"], df["y"])
    for c in ("o", "dir"):
        df[c] = np.where(left, (df[c] + 180.0) % 360.0, df[c])
    return df


def load_tracking_game(data_dir: Path, game_id: int, normalize: bool = True) -> pd.DataFrame:
    path = Path(data_dir) / "tracking" / f"tracking_{int(game_id)}.csv"
    if not path.exists():
        raise DataError(f"Tracking absent pour le match {game_id} ({path.name}).")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise DataError(f"Lecture impossible de {path.name} : {e}") from e
    miss = [c for c in ("gameId", "playId", "frameId", "x", "y", "s", "a") if c not in df.columns]
    if miss:
        raise DataError(f"Colonnes absentes dans {path.name} : {miss}")
    return normalize_tracking(df) if normalize else df


def vec(angle_deg: np.ndarray | float):
    """Angle NFL (0° = +y, sens horaire) -> vecteur unitaire (dx, dy)."""
    r = np.deg2rad(angle_deg)
    return np.sin(r), np.cos(r)
