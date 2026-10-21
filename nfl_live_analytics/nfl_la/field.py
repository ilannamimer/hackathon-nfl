"""Terrain NFL vu du dessus + animation Plotly d'une action (positions, vitesses, trajectoires, séparations, espace)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from .config import (C_ACCENT, C_BALL, C_BG, C_DEF, C_MUTED, C_OFF, C_PANEL, C_TURF, C_TURF2, C_TXT, FIELD_WID, FPS)
from .data import vec

EVENT_LABELS = {"ball_snap": "BALL SNAP", "autoevent_ballsnap": "BALL SNAP", "pass_forward": "PASS RELEASED",
                "autoevent_passforward": "PASS RELEASED", "pass_arrived": "PASS ARRIVED",
                "pass_outcome_caught": "CATCH", "pass_outcome_incomplete": "INCOMPLETE", "qb_sack": "SACK",
                "line_set": "LINE SET", "play_action": "PLAY ACTION", "man_in_motion": "MOTION",
                "run": "QB RUN", "tackle": "TACKLE", "first_contact": "CONTACT", "pass_tipped": "PASS TIPPED",
                "fumble": "FUMBLE", "handoff": "HANDOFF", "shift": "SHIFT"}


# ------------------------------------------------------------------ terrain statique
def _field_layout(fig: go.Figure, x0: float, x1: float, los: float | None, first_down: float | None) -> None:
    shapes = [dict(type="rect", x0=0, x1=120, y0=0, y1=FIELD_WID, fillcolor=C_TURF, line=dict(width=0), layer="below"),
              dict(type="rect", x0=0, x1=10, y0=0, y1=FIELD_WID, fillcolor="#173F55", line=dict(width=0), layer="below"),
              dict(type="rect", x0=110, x1=120, y0=0, y1=FIELD_WID, fillcolor="#4A1F2C", line=dict(width=0), layer="below")]
    for k in range(10, 111, 10 if False else 5):               # bandes de turf alternées toutes les 5 yds
        if (k // 5) % 2 == 0 and k < 110:
            shapes.append(dict(type="rect", x0=k, x1=k + 5, y0=0, y1=FIELD_WID, fillcolor=C_TURF2,
                               line=dict(width=0), layer="below"))
    for k in range(10, 111, 5):
        shapes.append(dict(type="line", x0=k, x1=k, y0=0, y1=FIELD_WID,
                           line=dict(color="rgba(255,255,255,%.2f)" % (0.75 if k % 10 == 0 else 0.35), width=1.6 if k % 10 == 0 else 1)))
    shapes += [dict(type="rect", x0=0, x1=120, y0=0, y1=FIELD_WID, line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)")]
    if los is not None:
        shapes.append(dict(type="line", x0=los, x1=los, y0=0, y1=FIELD_WID, line=dict(color="#3B82F6", width=3)))
    if first_down is not None and first_down < 110:
        shapes.append(dict(type="line", x0=first_down, x1=first_down, y0=0, y1=FIELD_WID, line=dict(color="#FFD60A", width=3)))
    # marques de hash (≈ 23,6 yds des lignes de touche)
    hx = np.arange(11, 110, 1.0)
    for y in (23.58, 29.72):
        shapes += [dict(type="line", x0=x, x1=x, y0=y - 0.4, y1=y + 0.4, line=dict(color="rgba(255,255,255,.55)", width=1)) for x in hx if x0 - 1 <= x <= x1 + 1]
    # numéros de yards (trace statique)
    nx = [x for x in range(20, 101, 10) if x0 - 5 <= x <= x1 + 5]
    lab = [str(min(x - 10, 110 - x)) for x in nx]
    fig.add_trace(go.Scatter(x=nx * 2, y=[7] * len(nx) + [FIELD_WID - 7] * len(nx), text=lab * 2, mode="text",
                             textfont=dict(color="rgba(255,255,255,.45)", size=26, family="Arial Black"),
                             hoverinfo="skip", showlegend=False))
    fig.update_layout(shapes=shapes)


# ------------------------------------------------------------------ espace contrôlé
def space_grid(px, py, vx, vy, off_mask, x0, x1, step=1.5, horizon=0.5):
    """Contrôle de l'espace : chaque cellule va à l'équipe dont un joueur l'atteindrait le plus vite
    (position projetée à +0,5 s). +1 = attaque, -1 = défense. INDICATEUR CALCULÉ, pas une probabilité."""
    gx = np.arange(x0, x1 + step, step); gy = np.arange(0, FIELD_WID + step, step)
    GX, GY = np.meshgrid(gx, gy)
    qx, qy = px + vx * horizon, py + vy * horizon
    d = np.hypot(GX[..., None] - qx, GY[..., None] - qy)
    near = d.argmin(axis=-1)
    z = np.where(off_mask[near], 1.0, -1.0)
    return gx, gy, z


def space_share(g_frame: pd.DataFrame, off_team: str, los: float, depth: float = 20.0) -> float | None:
    """% de la zone [LOS ; LOS+depth] contrôlée par l'attaque."""
    pl = g_frame[g_frame.nflId.notna()]
    if pl.empty:
        return None
    ux, uy = vec(pl["dir"].fillna(0).to_numpy())
    gx, gy, z = space_grid(pl.x.to_numpy(), pl.y.to_numpy(), ux * pl.s.to_numpy(), uy * pl.s.to_numpy(),
                           (pl.team == off_team).to_numpy(), los, los + depth)
    return float((z > 0).mean())


# ------------------------------------------------------------------ figure animée
def play_figure(g: pd.DataFrame, meta: pd.DataFrame, off_team: str, los: float | None, yards_to_go: float | None,
                sep_pair: tuple[float, float] | None = None, trail_ids: list[float] | None = None,
                show_vectors=True, show_space=False, start_frame: int | None = None, height=560) -> go.Figure:
    """g : tracking normalisé d'UNE action. meta : index nflId -> name, pos, jersey, role."""
    frames_ids = sorted(g.frameId.unique())
    by_f = {f: d for f, d in g.groupby("frameId")}
    ev = g[g.event.notna()].drop_duplicates("frameId").set_index("frameId")["event"].to_dict()
    # fenêtre de vue : autour du jeu
    pl = g[g.nflId.notna()]
    x0, x1 = max(0, pl.x.min() - 6), min(120, pl.x.max() + 8)
    x0, x1 = min(x0, (los or x0) - 12), max(x1, (los or x1) + 25)
    fd = (los + yards_to_go) if (los is not None and yards_to_go is not None) else None
    fig = go.Figure()
    _field_layout(fig, x0, x1, los, fd)
    trail_ids = trail_ids or []

    def trace_data(f):
        d = by_f[f]; p = d[d.nflId.notna()].copy(); b = d[d.nflId.isna()]
        p["side"] = np.where(p.team == off_team, "off", "def")
        m = meta.reindex(p.nflId.to_numpy())
        p["name"], p["pos"] = m["name"].to_numpy(), m["pos"].to_numpy()
        hov = (p.name.fillna("?") + " (" + p.pos.fillna("?") + ") #" + p.jerseyNumber.fillna(0).astype(int).astype(str)
               + "<br>vitesse " + p.s.round(1).astype(str) + " yd/s<br>accél. " + p.a.round(1).astype(str) + " yd/s²")
        out = {}
        for s in ("off", "def"):
            q = p[p.side == s]; h = hov[q.index]
            out[s] = dict(x=q.x, y=q.y, text=q.jerseyNumber.fillna(0).astype(int).astype(str), hovertext=h)
            ux, uy = vec(q["dir"].fillna(0).to_numpy())
            L = q.s.to_numpy() * 0.8                            # vecteur = déplacement projeté à +0,8 s
            xs = np.column_stack([q.x, q.x + ux * L, np.full(len(q), np.nan)]).ravel()
            ys = np.column_stack([q.y, q.y + uy * L, np.full(len(q), np.nan)]).ravel()
            out["v" + s] = dict(x=xs if show_vectors else [], y=ys if show_vectors else [])
        out["ball"] = dict(x=b.x, y=b.y)
        # trajectoires
        tx, ty = [], []
        for tid in trail_ids:
            h = g[(g.nflId == tid) & (g.frameId <= f)]
            tx += list(h.x) + [np.nan]; ty += list(h.y) + [np.nan]
        out["trail"] = dict(x=tx, y=ty)
        # séparation
        if sep_pair is not None:
            a, bb = p[p.nflId == sep_pair[0]], p[p.nflId == sep_pair[1]]
            if len(a) and len(bb):
                dist = float(np.hypot(a.x.iloc[0] - bb.x.iloc[0], a.y.iloc[0] - bb.y.iloc[0]))
                out["sep"] = dict(x=[a.x.iloc[0], bb.x.iloc[0]], y=[a.y.iloc[0], bb.y.iloc[0]])
                out["sepl"] = dict(x=[(a.x.iloc[0] + bb.x.iloc[0]) / 2], y=[(a.y.iloc[0] + bb.y.iloc[0]) / 2 + 1.6],
                                   text=[f"{dist:.1f} yd"])
        out.setdefault("sep", dict(x=[], y=[])); out.setdefault("sepl", dict(x=[], y=[], text=[]))
        if show_space:
            ux, uy = vec(p["dir"].fillna(0).to_numpy())
            gx, gy, z = space_grid(p.x.to_numpy(), p.y.to_numpy(), ux * p.s.to_numpy(), uy * p.s.to_numpy(),
                                   (p.side == "off").to_numpy(), x0, x1)
            out["heat"] = dict(x=gx, y=gy, z=z)
        else:
            out["heat"] = dict(x=[0], y=[0], z=[[0]])
        return out

    def build(f):
        t = trace_data(f)
        return [
            go.Heatmap(x=t["heat"]["x"], y=t["heat"]["y"], z=t["heat"]["z"], zmin=-1, zmax=1, opacity=0.32 if show_space else 0,
                       colorscale=[[0, C_DEF], [0.5, "rgba(0,0,0,0)"], [1, C_OFF]], showscale=False, hoverinfo="skip", zsmooth="best"),
            go.Scatter(x=t["voff"]["x"], y=t["voff"]["y"], mode="lines", line=dict(color=C_OFF, width=2.5), hoverinfo="skip"),
            go.Scatter(x=t["vdef"]["x"], y=t["vdef"]["y"], mode="lines", line=dict(color=C_DEF, width=2.5), hoverinfo="skip"),
            go.Scatter(x=t["trail"]["x"], y=t["trail"]["y"], mode="lines", line=dict(color=C_ACCENT, width=3, dash="dot"), hoverinfo="skip"),
            go.Scatter(x=t["sep"]["x"], y=t["sep"]["y"], mode="lines", line=dict(color="#FFFFFF", width=2, dash="dash"), hoverinfo="skip"),
            go.Scatter(x=t["off"]["x"], y=t["off"]["y"], mode="markers+text", text=t["off"]["text"], hovertext=t["off"]["hovertext"],
                       hoverinfo="text", textfont=dict(color="white", size=10),
                       marker=dict(size=21, color=C_OFF, line=dict(color="white", width=1.5))),
            go.Scatter(x=t["def"]["x"], y=t["def"]["y"], mode="markers+text", text=t["def"]["text"], hovertext=t["def"]["hovertext"],
                       hoverinfo="text", textfont=dict(color="white", size=10),
                       marker=dict(size=21, color=C_DEF, line=dict(color="white", width=1.5))),
            go.Scatter(x=t["ball"]["x"], y=t["ball"]["y"], mode="markers", hoverinfo="skip",
                       marker=dict(size=12, color=C_BALL, symbol="diamond", line=dict(color="black", width=1.5))),
            go.Scatter(x=t["sepl"]["x"], y=t["sepl"]["y"], mode="text", text=t["sepl"].get("text", []),
                       textfont=dict(color="white", size=15, family="Arial Black"), hoverinfo="skip"),
        ]

    start = start_frame if start_frame in frames_ids else frames_ids[0]
    for tr in build(start):
        fig.add_trace(tr)
    n_static = 1                                                  # trace des numéros de yards
    idx = list(range(n_static, n_static + 9))
    frames = []
    for f in frames_ids:
        label = EVENT_LABELS.get(ev.get(f, ""), "")
        frames.append(go.Frame(name=str(f), data=build(f), traces=idx, layout=go.Layout(annotations=[dict(
            x=0.5, y=1.0, xref="paper", yref="paper", yanchor="bottom", showarrow=False,
            text=f"<b>{label}</b>  ·  t = {(f - frames_ids[0]) / FPS:.1f}s" if label else f"t = {(f - frames_ids[0]) / FPS:.1f}s",
            font=dict(color=C_ACCENT if label else C_MUTED, size=15))])))
    fig.frames = frames
    steps = [dict(method="animate", label=(str(round((f - frames_ids[0]) / FPS, 1)) if i % 5 == 0 else ""), args=[[str(f)],
             dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))]) for i, f in enumerate(frames_ids)]
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=34, b=8), paper_bgcolor=C_BG, plot_bgcolor=C_TURF, font=dict(color=C_TXT),
        xaxis=dict(range=[x0, x1], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-1.5, FIELD_WID + 1.5], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
        showlegend=False, hoverlabel=dict(bgcolor=C_PANEL, font_color=C_TXT),
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.02, xanchor="left", yanchor="top", pad=dict(t=8), bgcolor=C_PANEL,
                          font=dict(color=C_TXT),
                          buttons=[dict(label="▶ Lecture", method="animate", args=[None, dict(frame=dict(duration=int(1000 / FPS), redraw=True),
                                        fromcurrent=True, transition=dict(duration=0))]),
                                   dict(label="▶ Ralenti ×0.4", method="animate", args=[None, dict(frame=dict(duration=int(2500 / FPS), redraw=True),
                                        fromcurrent=True, transition=dict(duration=0))]),
                                   dict(label="⏸ Pause", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False),
                                        transition=dict(duration=0))])])],
        sliders=[dict(active=frames_ids.index(start), steps=steps, x=0.28, len=0.7, y=-0.02, yanchor="top", pad=dict(t=8),
                      currentvalue=dict(prefix="t (s) : ", font=dict(color=C_MUTED)), font=dict(color=C_MUTED), bgcolor=C_PANEL,
                      activebgcolor=C_ACCENT)])
    return fig
