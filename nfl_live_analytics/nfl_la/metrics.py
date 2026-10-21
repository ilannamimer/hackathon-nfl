"""Indicateurs calculés à partir du tracking.

Tout ce qui est ici est un INDICATEUR CALCULÉ (déterministe, formule explicite) — pas une prédiction.
Les probabilités de modèles sont produites séparément dans `models.py`.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import FPS
from .data import vec

SNAP_EVENTS = {"ball_snap", "autoevent_ballsnap"}
RELEASE_EVENTS = {"pass_forward", "autoevent_passforward"}


# ----------------------------------------------------------------------------- utilitaires
def event_frame(g: pd.DataFrame, names: set[str]) -> int | None:
    """Premier frameId portant l'un des événements `names` (None si absent)."""
    f = g.loc[g["event"].isin(names), "frameId"]
    return int(f.min()) if len(f) else None


def pivots(g: pd.DataFrame, cols=("x", "y", "s", "dir")) -> dict[str, pd.DataFrame]:
    """frameId × nflId pour chaque colonne (ballon exclu)."""
    pl = g[g["nflId"].notna()]
    return {c: pl.pivot(index="frameId", columns="nflId", values=c) for c in cols}


def ball_track(g: pd.DataFrame) -> pd.DataFrame:
    return g[g["nflId"].isna()].set_index("frameId")[["x", "y", "s"]].sort_index()


def min_dist(px: np.ndarray, py: np.ndarray, qx: np.ndarray, qy: np.ndarray) -> np.ndarray:
    """Pour chaque point P, distance au point Q le plus proche (matrices n_p x n_q)."""
    if len(qx) == 0:
        return np.full(len(px), np.nan)
    d = np.hypot(px[:, None] - qx[None, :], py[:, None] - qy[None, :])
    return d.min(axis=1)


# ----------------------------------------------------------------------------- séparation
def separation_timeline(g: pd.DataFrame, receiver_id: float, defense_ids: list[float]) -> pd.DataFrame:
    """Distance receveur → défenseur le plus proche, image par image (indicateur calculé)."""
    pv = pivots(g, ("x", "y", "s"))
    rx, ry = pv["x"][receiver_id], pv["y"][receiver_id]
    dx = pv["x"][defense_ids].sub(rx, axis=0)
    dy = pv["y"][defense_ids].sub(ry, axis=0)
    dist = np.hypot(dx, dy)
    out = pd.DataFrame({"frameId": dist.index, "sep": dist.min(axis=1).to_numpy(),
                        "nearest_def": dist.idxmin(axis=1).to_numpy(),
                        "speed": pv["s"][receiver_id].to_numpy()})
    return out


# ----------------------------------------------------------------------------- pression sur le QB
def pressure_index(d_rush: np.ndarray, closing: np.ndarray) -> np.ndarray:
    """PRESSURE INDEX (0-100) = 100 * (0.7 * proximité + 0.3 * vitesse de rapprochement).

    proximité = clip((6 - d)/6, 0, 1) : 0 à ≥6 yds, 1 au contact ;
    vitesse de rapprochement = clip(closing/6, 0, 1) (yds/s). Pondérations choisies a priori
    (la distance est le signal dominant) ; validité vérifiée a posteriori vs pff_hurry/hit/sack (AUC).
    """
    prox = np.clip((6.0 - d_rush) / 6.0, 0, 1)
    clo = np.clip(closing / 6.0, 0, 1)
    return 100.0 * (0.7 * prox + 0.3 * clo)


def qb_pressure_timeline(g: pd.DataFrame, qb_id: float, rusher_ids: list[float]) -> pd.DataFrame:
    pv = pivots(g, ("x", "y"))
    if qb_id not in pv["x"].columns or not rusher_ids:
        return pd.DataFrame(columns=["frameId", "d_rush", "closing", "pressure"])
    rids = [r for r in rusher_ids if r in pv["x"].columns]
    dx = pv["x"][rids].sub(pv["x"][qb_id], axis=0)
    dy = pv["y"][rids].sub(pv["y"][qb_id], axis=0)
    d = np.hypot(dx, dy).min(axis=1)
    lag = 3  # 0,3 s
    closing = ((d.shift(lag) - d) / (lag / FPS)).fillna(0).clip(lower=0)
    return pd.DataFrame({"frameId": d.index, "d_rush": d.to_numpy(), "closing": closing.to_numpy(),
                         "pressure": pressure_index(d.to_numpy(), closing.to_numpy())})


# ----------------------------------------------------------------------------- cible estimée
def estimate_target(ball: pd.DataFrame, rel: int, pv: dict, route_ids: list, ) -> tuple[float | None, float]:
    """Heuristique : le receveur dont la position PROJETÉE est la mieux alignée avec la direction
    initiale du ballon (images rel+1…rel+4). Retourne (nflId, écart angulaire en degrés).

    Les données ne contiennent PAS le receveur ciblé : c'est une estimation, validée plus bas
    sur les passes complétées (voir build_features.py).
    """
    try:
        b0, b1 = ball.loc[rel], ball.loc[rel + 4]
    except KeyError:
        return None, np.nan
    vx, vy = b1["x"] - b0["x"], b1["y"] - b0["y"]
    n = np.hypot(vx, vy)
    if n < 1.0 or not route_ids:
        return None, np.nan
    vx, vy = vx / n, vy / n
    ball_speed = max(float(ball.loc[rel + 4, "s"]), 8.0)
    best, best_ang = None, 1e9
    for rid in route_ids:
        if rid not in pv["x"].columns or rel not in pv["x"].index:
            continue
        px, py = pv["x"].at[rel, rid], pv["y"].at[rel, rid]
        sp, dr = pv["s"].at[rel, rid], pv["dir"].at[rel, rid]
        t = np.hypot(px - b0["x"], py - b0["y"]) / ball_speed
        ux, uy = vec(dr if np.isfinite(dr) else 0.0)
        qx, qy = px + ux * sp * t, py + uy * sp * t          # position projetée à l'arrivée
        ax, ay = qx - b0["x"], qy - b0["y"]
        an = np.hypot(ax, ay)
        if an < 1e-6:
            continue
        ang = np.degrees(np.arccos(np.clip((ax * vx + ay * vy) / an, -1, 1)))
        if ang < best_ang:
            best, best_ang = rid, ang
    return best, best_ang


# ----------------------------------------------------------------------------- features par action
def play_features(g: pd.DataFrame, roles: pd.DataFrame, play: pd.Series) -> dict:
    """Features d'une action, calculées à l'instant du lancer (aucune info sur le résultat)."""
    out = {"gameId": int(play.gameId), "playId": int(play.playId)}
    snap, rel = event_frame(g, SNAP_EVENTS), event_frame(g, RELEASE_EVENTS)
    ball, pv = ball_track(g), pivots(g)
    role = roles.set_index("nflId")["pff_role"].to_dict()
    ids = list(pv["x"].columns)
    qb = next((i for i in ids if role.get(i) == "Pass"), None)
    rush = [i for i in ids if role.get(i) == "Pass Rush"]
    block = [i for i in ids if role.get(i) == "Pass Block"]
    route = [i for i in ids if role.get(i) == "Pass Route"]
    cover = [i for i in ids if role.get(i) == "Coverage"]
    team = g[g["nflId"].notna()].drop_duplicates("nflId").set_index("nflId")["team"].to_dict()
    defense = [i for i in ids if team.get(i) == play.defensiveTeam]

    out.update(n_rush=len(rush), n_block=len(block), n_routes=len(route), n_cover=len(cover),
               snap_frame=snap, release_frame=rel)
    if snap is not None and snap in ball.index:
        out["los_x"] = float(ball.at[snap, "x"])
        out["yards_to_goal"] = 110.0 - out["los_x"]
        out["los_y"] = float(ball.at[snap, "y"])
    # pression max entre snap et lancer (ou fin de l'action pour les sacks)
    if qb is not None and rush:
        tl = qb_pressure_timeline(g, qb, rush)
        stop = rel if rel is not None else tl.frameId.max()
        w = tl[(tl.frameId >= (snap or 1)) & (tl.frameId <= stop)]
        if len(w):
            out["max_pressure"] = float(w.pressure.max())
            out["min_rush_dist_pre"] = float(w.d_rush.min())
    if rel is None or qb is None or rel not in pv["x"].index:
        return out

    out["time_to_throw"] = (rel - (snap or rel)) / FPS
    X, Y, S = pv["x"].loc[rel], pv["y"].loc[rel], pv["s"].loc[rel]
    out["qb_speed"] = float(S[qb])
    out["qb_depth"] = float(out.get("los_x", np.nan) - X[qb]) if "los_x" in out else np.nan
    if rush:
        dr = np.hypot(X[rush].to_numpy() - X[qb], Y[rush].to_numpy() - Y[qb])
        ds = np.sort(dr)
        out["rush_min_dist"] = float(ds[0])
        out["rush_mean2_dist"] = float(ds[:2].mean())
        out["n_rush_within5"] = int((dr < 5).sum())
        tl = qb_pressure_timeline(g, qb, rush).set_index("frameId")
        if rel in tl.index:
            out["pressure_at_release"] = float(tl.at[rel, "pressure"])
    # séparations de tous les receveurs au lancer
    if route and defense:
        sep = min_dist(X[route].to_numpy(), Y[route].to_numpy(), X[defense].to_numpy(), Y[defense].to_numpy())
        out["best_sep"], out["mean_sep"] = float(np.nanmax(sep)), float(np.nanmean(sep))
        out["n_open3"] = int((sep >= 3).sum())
        # cible estimée
        tgt, ang = estimate_target(ball, rel, pv, route)
        out["target_id"], out["target_angle"] = tgt, ang
        if tgt is not None:
            tx, ty = X[tgt], Y[tgt]
            out["target_sep"] = float(min_dist(np.array([tx]), np.array([ty]), X[defense].to_numpy(),
                                               Y[defense].to_numpy())[0])
            out["target_dist"] = float(np.hypot(tx - X[qb], ty - Y[qb]))
            out["target_depth"] = float(tx - out.get("los_x", tx))
            out["target_speed"] = float(S[tgt])
    return out


# ----------------------------------------------------------------------------- explosivité joueur
def player_play_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Métriques par (action, joueur) pour tout un match normalisé, de la mise en jeu à la fin du suivi.

    vmax  : vitesse maximale (yds/s)
    amax  : accélération maximale (yds/s²) — quantile 99 % des valeurs positives pour limiter le bruit
    cuts  : nombre de changements de direction francs (|Δdir| ≥ 45° en 0,3 s à vitesse ≥ 2 yds/s)
    """
    d = df[df["nflId"].notna()].copy()
    snap = (df[df["event"].isin(SNAP_EVENTS) & df["nflId"].isna()].groupby("playId")["frameId"].min()
            .rename("snap"))
    d = d.merge(snap, on="playId", how="left")
    d = d[d["frameId"] >= d["snap"].fillna(1)].sort_values(["playId", "nflId", "frameId"])
    grp = d.groupby(["playId", "nflId"])
    lag = 3
    dd = (d["dir"] - grp["dir"].shift(lag) + 180.0) % 360.0 - 180.0   # écart d'angle signé, [-180,180)
    sharp = (dd.abs() >= 45) & (d["s"] >= 2.0)
    # on ne compte qu'un « cut » par séquence continue de frames franches
    new = sharp & ~(sharp.groupby([d["playId"], d["nflId"]]).shift(1, fill_value=False))
    d["_cut"] = new.astype(int)
    d["_apos"] = d["a"].where(d["a"] > 0)
    agg = grp.agg(vmax=("s", "max"), dist=("dis", "sum"), cuts=("_cut", "sum"),
                  frames=("frameId", "count"))
    agg["amax"] = grp["_apos"].quantile(0.99)
    return agg.reset_index()
