"""Terrain NFL 3D interactif (Plotly.js dans un composant HTML) avec fiche joueur au clic.

Tout est calculé côté navigateur à partir des données embarquées : animation fluide, clic sur un joueur
-> panneau latéral (identité, stats instantanées, stats de l'action, indice d'explosivité, courbe de vitesse),
sans recharger la page Streamlit.

Convention : coordonnées normalisées (l'attaque va vers +x). Hauteur des joueurs = représentation visuelle
uniquement (le tracking ne contient pas de coordonnée verticale).
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import metrics, teams
from .config import C_NAVY, C_TURF, C_TURF2, FIELD_WID
from .field import EVENT_LABELS

PLOTLY_JS = "https://cdn.plot.ly/plotly-2.35.2.min.js"
STEM_H = 2.0
IDX = dict(heat=8, shadow=9, vec_off=10, vec_def=11, trail=12, sep=13, sepl=14, stem_off=15, stem_def=16,
           pin_off=17, pin_def=18, ball=19, sel=20)
CAMERAS = {
    "broadcast": dict(eye=dict(x=-1.25, y=-0.95, z=0.85), center=dict(x=0, y=0, z=-0.1), up=dict(x=0, y=0, z=1)),
    "aerienne": dict(eye=dict(x=0.0, y=0.001, z=1.9), center=dict(x=0, y=0, z=-0.2), up=dict(x=0, y=1, z=0)),
    "arriere": dict(eye=dict(x=-1.7, y=0.0, z=0.5), center=dict(x=0, y=0, z=-0.1), up=dict(x=0, y=0, z=1)),
    "cote": dict(eye=dict(x=0.0, y=-1.7, z=0.55), center=dict(x=0, y=0, z=-0.1), up=dict(x=0, y=0, z=1)),
}
ROLE_FR = {"Pass": "Passeur", "Pass Route": "Receveur (route)", "Pass Block": "Bloqueur", "Pass Rush": "Rusher",
           "Coverage": "Couverture"}


def _clean(a) -> list:
    return [None if (v is None or not np.isfinite(v)) else round(float(v), 2) for v in a]


def _sanitize(o):
    """NaN/inf/numpy -> types JSON valides (None pour les valeurs manquantes)."""
    if isinstance(o, dict):
        return {str(k): _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    return o


def static_figure(x0, x1, los, ytg, off_abbr, def_abbr, off_c, def_c, gx, gy) -> go.Figure:
    """Terrain 3D (mesh + lignes) et traces dynamiques vides (remplies par le JS)."""
    fig = go.Figure()
    # --- surface du terrain : un rectangle (2 triangles) par bande de couleur
    bands = [(0, 10, off_c), (110, 120, def_c)] + [(k, k + 5, C_TURF if (k // 5) % 2 else C_TURF2) for k in range(10, 110, 5)]
    X, Y, Z, I, J, K, FC = [], [], [], [], [], [], []
    for a, b, col in bands:
        a, b = max(a, x0), min(b, x1)
        if b <= a:
            continue
        n = len(X)
        X += [a, b, b, a]; Y += [0, 0, FIELD_WID, FIELD_WID]; Z += [0] * 4
        I += [n, n]; J += [n + 1, n + 2]; K += [n + 2, n + 3]; FC += [col, col]
    fig.add_trace(go.Mesh3d(x=X, y=Y, z=Z, i=I, j=J, k=K, facecolor=FC, flatshading=True, hoverinfo="skip",
                            lighting=dict(ambient=1, diffuse=0, specular=0, roughness=1), showscale=False))
    # --- lignes
    def seg(xs):                                    # lignes verticales (largeur du terrain) aux abscisses xs
        lx, ly, lz = [], [], []
        for x in xs:
            lx += [x, x, None]; ly += [0, FIELD_WID, None]; lz += [0.03, 0.03, None]
        return lx, ly, lz
    for xs, w, op in (([k for k in range(10, 111, 5) if k % 10 and x0 <= k <= x1], 2, .45),
                      ([k for k in range(10, 111, 10) if x0 <= k <= x1], 3, .85)):
        lx, ly, lz = seg(xs)
        fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode="lines", line=dict(color=f"rgba(255,255,255,{op})", width=w), hoverinfo="skip"))
    nx = [x for x in range(20, 101, 10) if x0 <= x <= x1]
    fig.add_trace(go.Scatter3d(x=nx * 2, y=[7] * len(nx) + [FIELD_WID - 7] * len(nx), z=[0.03] * (2 * len(nx)),
                               text=[str(min(x - 10, 110 - x)) for x in nx] * 2, mode="text",
                               textfont=dict(color="rgba(255,255,255,.55)", size=20), hoverinfo="skip"))
    ez = [(5, off_abbr), (115, def_abbr)]
    ez = [(x, a) for x, a in ez if x0 <= x <= x1 and a]
    fig.add_trace(go.Scatter3d(x=[e[0] for e in ez], y=[FIELD_WID / 2] * len(ez), z=[0.03] * len(ez), text=[e[1] for e in ez],
                               mode="text", textfont=dict(color="rgba(255,255,255,.85)", size=26), hoverinfo="skip"))
    for xv, col in ((los, "#3B82F6"), ((los + ytg) if (los is not None and ytg is not None) else None, "#FFD60A")):
        vis = xv is not None and x0 <= xv <= x1
        fig.add_trace(go.Scatter3d(x=[xv, xv] if vis else [], y=[0, FIELD_WID] if vis else [], z=[0.05, 0.05] if vis else [],
                                   mode="lines", line=dict(color=col, width=8), hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=[x0, x1, None, x0, x1], y=[0, 0, None, FIELD_WID, FIELD_WID], z=[0.03] * 5, mode="lines",
                               line=dict(color="white", width=5), hoverinfo="skip"))
    assert len(fig.data) == IDX["heat"], len(fig.data)
    # --- dynamiques (placeholders)
    fig.add_trace(go.Surface(x=gx, y=gy, z=[[0.06] * len(gx) for _ in gy], surfacecolor=[[0] * len(gx) for _ in gy], cmin=-1, cmax=1,
                             colorscale=[[0, def_c], [0.5, C_TURF], [1, off_c]], showscale=False, opacity=0.45, hoverinfo="skip",
                             visible=False, lighting=dict(ambient=1, diffuse=0, specular=0)))
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="markers", marker=dict(size=13, color="rgba(0,0,0,.32)"), hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines", line=dict(color="#FFFFFF", width=5), hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines", line=dict(color="#FFE08A", width=5), hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines", line=dict(color="#FFD60A", width=7), hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines", line=dict(color="#FFFFFF", width=5, dash="dash"), hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="text", textfont=dict(color="#FFFFFF", size=16), hoverinfo="skip"))
    for col in (off_c, def_c):
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines", line=dict(color=col, width=7), hoverinfo="skip"))
    for col in (off_c, def_c):
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="markers+text", text=[], hovertext=[], hoverinfo="text", textposition="middle center",
                                   textfont=dict(color=teams.text_on(col), size=10),
                                   marker=dict(size=15, color=col, line=dict(color="white", width=2))))
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="markers", hoverinfo="skip",
                               marker=dict(size=9, color="#FFC933", symbol="diamond", line=dict(color="black", width=2))))
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="markers", hoverinfo="skip",
                               marker=dict(size=28, color="rgba(0,0,0,0)", symbol="circle-open", line=dict(color="#FFD60A", width=6))))
    fig.update_layout(
        showlegend=False, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#EEF1F5",
        scene=dict(xaxis=dict(visible=False, range=[x0, x1]), yaxis=dict(visible=False, range=[0, FIELD_WID]),
                   zaxis=dict(visible=False, range=[0, 6]), bgcolor="#EEF1F5", aspectmode="manual",
                   aspectratio=dict(x=(x1 - x0) / FIELD_WID, y=1, z=0.2), camera=CAMERAS["broadcast"]),
        hoverlabel=dict(bgcolor="#FFFFFF", font=dict(color="#111827", size=12)))
    return fig


def build_payload(g, meta, role, lined, PLAYERS, PI, P, rcv, qb, los, ytg, snap, rel) -> tuple[dict, float, float]:
    frames = sorted(g.frameId.unique())
    pl = g[g.nflId.notna()]
    piv = {c: pl.pivot(index="frameId", columns="nflId", values=c).reindex(frames) for c in ("x", "y", "s", "a", "dir", "o")}
    ball = g[g.nflId.isna()].set_index("frameId").reindex(frames)
    pm = metrics.player_play_metrics(g).set_index("nflId")
    jersey = pl.drop_duplicates("nflId").set_index("nflId")["jerseyNumber"].to_dict()
    team = pl.drop_duplicates("nflId").set_index("nflId")["team"].to_dict()
    PI = PI.copy()
    PI["rank"] = PI.groupby("group")["EI"].rank(ascending=False, method="min")
    PI["n_group"] = PI.groupby("group")["EI"].transform("size")
    PIx = PI.set_index("nflId")
    f0 = frames[0]
    players = []
    for nid in piv["x"].columns:
        info = PLAYERS.loc[nid] if nid in PLAYERS.index else None
        side = "off" if team.get(nid) == P.possessionTeam else "def"
        d = dict(id=int(nid), jersey=int(jersey[nid]) if jersey.get(nid) == jersey.get(nid) else None, team=team.get(nid), side=side,
                 name=meta.name.get(nid, f"#{int(nid)}"), pos=(info.officialPosition if info is not None else meta.pos.get(nid, "?")),
                 lined=lined.get(nid, ""), role=ROLE_FR.get(role.get(nid, ""), role.get(nid, "")),
                 height=(info.height if info is not None else None), weight=(int(info.weight) if info is not None and info.weight == info.weight else None),
                 college=(info.collegeName if info is not None else None),
                 x=_clean(piv["x"][nid]), y=_clean(piv["y"][nid]), s=_clean(piv["s"][nid]), a=_clean(piv["a"][nid]),
                 dir=_clean(piv["dir"][nid]), o=_clean(piv["o"][nid]),
                 pm=({k: (None if not np.isfinite(pm.at[nid, k]) else round(float(pm.at[nid, k]), 2)) for k in ("vmax", "amax", "cuts", "dist")}
                     if nid in pm.index else None),
                 season=None)
        if nid in PIx.index:
            r = PIx.loc[nid]
            d["season"] = dict(EI=round(float(r.EI), 1), group=r.group, rank=int(r["rank"]), n=int(r.n_group), plays=int(r.plays),
                               vmax=round(float(r.vmax), 2), amax=round(float(r.amax), 2), cuts=round(float(r.cuts), 2))
        players.append(d)
    ev = g[g.event.notna()].drop_duplicates("frameId").set_index("frameId")["event"].to_dict()
    events = {str(frames.index(f)): EVENT_LABELS.get(e, e) for f, e in ev.items() if f in frames}
    off_c, def_c = teams.matchup_colors(P.possessionTeam, P.defensiveTeam)
    allx = piv["x"].to_numpy(); allx = allx[np.isfinite(allx)]
    x0 = float(max(0, min(allx.min() - 6, (los or allx.min()) - 12)))
    x1 = float(min(120, max(allx.max() + 8, (los or allx.max()) + 25)))
    payload = dict(
        frames=[int(f) for f in frames], t=[round((f - f0) / 10, 1) for f in frames], events=events,
        ball=dict(x=_clean(ball["x"]), y=_clean(ball["y"])), players=players,
        ctx=dict(los=los, ytg=ytg, offAbbr=P.possessionTeam, defAbbr=P.defensiveTeam, offColor=off_c, defColor=def_c,
                 offName=teams.name(P.possessionTeam), defName=teams.name(P.defensiveTeam),
                 offLogo=teams.logo_url(P.possessionTeam), defLogo=teams.logo_url(P.defensiveTeam),
                 qb=None if qb is None else int(qb), rcv=None if rcv is None else int(rcv),
                 target=None if not (P.target_final == P.target_final) else int(P.target_final),
                 snapIdx=None if snap is None else frames.index(snap), relIdx=None if rel is None else frames.index(rel),
                 x0=x0, x1=x1, stemH=STEM_H, idx=IDX))
    return payload, x0, x1


def build_html(g, meta, role, lined, PLAYERS, PI, P, rcv, qb, los, ytg, snap, rel, height: int = 760) -> str:
    payload, x0, x1 = build_payload(g, meta, role, lined, PLAYERS, PI, P, rcv, qb, los, ytg, snap, rel)
    step = 1.5
    gx = [round(v, 2) for v in np.arange(x0, x1 + 0.01, step)]
    gy = [round(v, 2) for v in np.arange(0, FIELD_WID + 0.01, step)]
    payload["grid"] = dict(gx=gx, gy=gy)
    off_c, def_c = payload["ctx"]["offColor"], payload["ctx"]["defColor"]
    fig = static_figure(x0, x1, los, ytg, P.possessionTeam, P.defensiveTeam, off_c, def_c, gx, gy)
    fig_json = json.loads(fig.to_json())
    html = (TEMPLATE.replace("__PLOTLY__", PLOTLY_JS).replace("__NAVY__", C_NAVY).replace("__H__", str(height))
            .replace("__CAMS__", json.dumps(CAMERAS)).replace("__FIG__", json.dumps({"data": fig_json["data"], "layout": fig_json["layout"]}, separators=(",", ":")))
            .replace("__DATA__", json.dumps(_sanitize(payload), separators=(",", ":"), allow_nan=False)))
    return html


TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=Barlow:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="__PLOTLY__"></script>
<style>
*{box-sizing:border-box} body{margin:0;font-family:'Barlow',Arial,sans-serif;color:#111827;background:#fff}
.wrap{display:flex;gap:12px;height:__H__px}
.left{flex:1;min-width:0;display:flex;flex-direction:column;gap:8px}
#plot{flex:1;min-height:0;border:1px solid #DDE2E8;border-radius:4px;background:#EEF1F5}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.btn{font:700 14px 'Barlow Condensed',sans-serif;letter-spacing:.05em;text-transform:uppercase;border:1px solid #013369;background:#fff;color:#013369;
  padding:6px 14px;border-radius:3px;cursor:pointer} .btn:hover{background:#E3ECF8} .btn.on{background:#013369;color:#fff}
.btn.red{background:#D50A0A;border-color:#D50A0A;color:#fff}
.grp{display:flex;gap:4px;align-items:center;padding-right:10px;border-right:1px solid #DDE2E8}
.grp:last-child{border:0}
.lbl{font:600 11px 'Barlow',sans-serif;color:#5B6770;letter-spacing:.08em;text-transform:uppercase;margin-right:4px}
input[type=range]{flex:1;min-width:180px;accent-color:#D50A0A}
#tlabel{font:800 18px 'Barlow Condensed',sans-serif;color:#013369;min-width:210px;text-transform:uppercase;letter-spacing:.04em}
.panel{width:330px;flex:none;border:1px solid #DDE2E8;border-radius:4px;overflow:auto;background:#fff;box-shadow:0 2px 8px rgba(1,51,105,.08)}
.ph{padding:14px 16px;color:#fff;display:flex;gap:12px;align-items:center}
.ph img{height:52px;width:52px;object-fit:contain;background:rgba(255,255,255,.92);border-radius:50%;padding:4px}
.ph .num{font:800 40px/1 'Barlow Condensed',sans-serif;opacity:.9}
.ph .nm{font:800 24px/1.05 'Barlow Condensed',sans-serif;text-transform:uppercase}
.ph .sub{font:500 12px 'Barlow',sans-serif;opacity:.9;letter-spacing:.05em;text-transform:uppercase}
.sec{padding:10px 16px;border-top:1px solid #EDF0F4}
.sec h4{margin:0 0 6px 0;font:800 14px 'Barlow Condensed',sans-serif;letter-spacing:.1em;color:#013369;text-transform:uppercase}
.kv{display:flex;justify-content:space-between;font-size:13px;padding:2.5px 0}
.kv span:first-child{color:#5B6770}.kv span:last-child{font-weight:700}
.big{display:flex;gap:10px}.big div{flex:1;background:#F2F4F7;border-radius:3px;padding:6px 8px}
.big .v{font:800 26px/1 'Barlow Condensed',sans-serif;color:#111827}.big .k{font:600 10px 'Barlow';color:#5B6770;letter-spacing:.08em;text-transform:uppercase}
.tag{display:inline-block;font:700 10px 'Barlow';padding:2px 7px;border-radius:2px;background:#FFF0D1;color:#8A5A00;letter-spacing:.06em;margin-left:6px;vertical-align:middle}
.empty{padding:30px 20px;color:#5B6770;font-size:14px;line-height:1.5}
.hint{font-size:12px;color:#5B6770}
canvas{width:100%;height:84px;display:block}
</style></head><body>
<div class="wrap">
 <div class="left">
  <div class="bar">
   <div class="grp"><button class="btn red" id="play">Lecture</button><button class="btn" id="slow">Ralenti</button>
     <button class="btn" id="gosnap">Snap</button><button class="btn" id="gorel">Lancer</button></div>
   <div class="grp"><span class="lbl">Vue</span><button class="btn on" data-cam="broadcast">Broadcast</button><button class="btn" data-cam="aerienne">Aérienne</button>
     <button class="btn" data-cam="arriere">Arrière</button><button class="btn" data-cam="cote">Côté</button></div>
   <div class="grp"><button class="btn on" id="tvec">Vecteurs</button><button class="btn" id="tspace">Espace</button><button class="btn on" id="ttrail">Trajectoires</button></div>
  </div>
  <div id="plot"></div>
  <div class="bar"><span id="tlabel"></span><input type="range" id="slider" min="0" max="0" value="0"></div>
  <div class="hint">Glisser pour tourner · molette pour zoomer · clic sur un joueur pour afficher sa fiche. La hauteur des joueurs est purement visuelle (non mesurée).</div>
 </div>
 <div class="panel" id="panel"></div>
</div>
<script>
const D=__DATA__, FIG=__FIG__, CAMS=__CAMS__, C=D.ctx, IDX=C.idx;
const N=D.frames.length, H=C.stemH;
const P=D.players, byId={}; P.forEach(p=>byId[p.id]=p);
const off=P.filter(p=>p.side==='off'), def=P.filter(p=>p.side==='def');
const gd=document.getElementById('plot');
let cur=(C.relIdx!=null?C.relIdx:0), sel=(C.target!=null&&byId[C.target]?C.target:(C.rcv!=null?C.rcv:null));
let timer=null, rate=1, showVec=true, showSpace=false, showTrail=true;
const ok=v=>v!==null&&v!==undefined&&isFinite(v);
const f1=v=>ok(v)?v.toFixed(1):'—', f2=v=>ok(v)?v.toFixed(2):'—';
const YD2KMH=3.2918;
const vecOf=d=>[Math.sin(d*Math.PI/180),Math.cos(d*Math.PI/180)];

function nearestOpp(p,i){let best=null,bd=1e9;const pool=p.side==='off'?def:off;
  for(const q of pool){if(!ok(q.x[i])||!ok(p.x[i]))continue;const d=Math.hypot(q.x[i]-p.x[i],q.y[i]-p.y[i]);if(d<bd){bd=d;best=q;}}
  return best?{p:best,d:bd}:null;}
function dist(a,b,i){return(ok(a.x[i])&&ok(b.x[i]))?Math.hypot(a.x[i]-b.x[i],a.y[i]-b.y[i]):null;}

function stems(list,i){const X=[],Y=[],Z=[];for(const p of list){if(!ok(p.x[i]))continue;X.push(p.x[i],p.x[i],null);Y.push(p.y[i],p.y[i],null);Z.push(0.05,H,null);}return[X,Y,Z];}
function pins(list,i){const X=[],Y=[],Z=[],T=[],HV=[];
  for(const p of list){X.push(ok(p.x[i])?p.x[i]:null);Y.push(ok(p.y[i])?p.y[i]:null);Z.push(ok(p.x[i])?H:null);T.push(p.jersey!=null?String(p.jersey):'');
    HV.push(p.name+' ('+p.pos+') #'+p.jersey+'<br>vitesse '+f1(p.s[i])+' yd/s<br>clic : fiche joueur');}
  return[X,Y,Z,T,HV];}
function vectors(list,i){const X=[],Y=[],Z=[];if(!showVec)return[X,Y,Z];
  for(const p of list){if(!ok(p.x[i])||!ok(p.dir[i])||!ok(p.s[i]))continue;const[v,w]=vecOf(p.dir[i]);const L=p.s[i]*0.8;
    X.push(p.x[i],p.x[i]+v*L,null);Y.push(p.y[i],p.y[i]+w*L,null);Z.push(0.08,0.08,null);}return[X,Y,Z];}
function trails(i){const X=[],Y=[],Z=[];if(!showTrail)return[X,Y,Z];const ids=[...new Set([C.qb,C.rcv,sel].filter(v=>v!=null))];
  for(const id of ids){const p=byId[id];if(!p)continue;for(let k=0;k<=i;k++){if(ok(p.x[k])){X.push(p.x[k]);Y.push(p.y[k]);Z.push(0.1);}}X.push(null);Y.push(null);Z.push(null);}return[X,Y,Z];}
const E6=()=>[[],[],[],[],[],[]];
function sepLine(i){const id=sel!=null?sel:C.rcv;const p=id!=null?byId[id]:null;if(!p||p.id===C.qb)return E6();
  const n=nearestOpp(p,i);if(!n)return E6();
  return[[p.x[i],n.p.x[i],null],[p.y[i],n.p.y[i],null],[1.0,1.0,null],[(p.x[i]+n.p.x[i])/2],[(p.y[i]+n.p.y[i])/2],[n.d]];}

function heat(i){const gx=D.grid.gx,gy=D.grid.gy;const pts=[];
  for(const p of P){if(!ok(p.x[i]))continue;const sp=ok(p.s[i])?p.s[i]:0,[v,w]=ok(p.dir[i])?vecOf(p.dir[i]):[0,0];pts.push([p.x[i]+v*sp*0.5,p.y[i]+w*sp*0.5,p.side==='off'?1:-1]);}
  return gy.map(y=>gx.map(x=>{let b=1e9,s=0;for(const q of pts){const d=(q[0]-x)*(q[0]-x)+(q[1]-y)*(q[1]-y);if(d<b){b=d;s=q[2];}}return s;}));}

function updateScene(){const i=cur;
  const so=stems(off,i),sd=stems(def,i),po=pins(off,i),pd=pins(def,i),vo=vectors(off,i),vd=vectors(def,i),tr=trails(i),sp=sepLine(i);
  const shX=[],shY=[],shZ=[];for(const p of P){if(ok(p.x[i])){shX.push(p.x[i]);shY.push(p.y[i]);shZ.push(0.06);}}
  const bx=ok(D.ball.x[i])?[D.ball.x[i]]:[],by=ok(D.ball.y[i])?[D.ball.y[i]]:[];
  const sp_=sel!=null?byId[sel]:null;const sx=sp_&&ok(sp_.x[i])?[sp_.x[i]]:[],sy=sp_&&ok(sp_.x[i])?[sp_.y[i]]:[],sz=sp_&&ok(sp_.x[i])?[H]:[];
  const order=[IDX.shadow,IDX.vec_off,IDX.vec_def,IDX.trail,IDX.sep,IDX.sepl,IDX.stem_off,IDX.stem_def,IDX.pin_off,IDX.pin_def,IDX.ball,IDX.sel];
  const xs=[shX,vo[0],vd[0],tr[0],sp[0],sp[3],so[0],sd[0],po[0],pd[0],bx,sx];
  const ys=[shY,vo[1],vd[1],tr[1],sp[1],sp[4],so[1],sd[1],po[1],pd[1],by,sy];
  const zs=[shZ,vo[2],vd[2],tr[2],sp[2],sp[3].map(()=>1.6),so[2],sd[2],po[2],pd[2],bx.map(()=>0.25),sz];
  Plotly.restyle(gd,{x:xs,y:ys,z:zs},order);
  Plotly.restyle(gd,{text:[po[3],pd[3],sp[5].map(d=>d.toFixed(1)+' yd')],hovertext:[po[4],pd[4],[]]},[IDX.pin_off,IDX.pin_def,IDX.sepl]);
  if(showSpace)Plotly.restyle(gd,{surfacecolor:[heat(i)]},[IDX.heat]);
  document.getElementById('slider').value=i;
  const ev=D.events[String(i)];document.getElementById('tlabel').textContent='t = '+D.t[i].toFixed(1)+' s'+(ev?'  ·  '+ev:'');
  renderPanel();}

function spark(p){const cv=document.getElementById('spark');if(!cv)return;const w=cv.width=cv.clientWidth*2||600,h=cv.height=168;const g=cv.getContext('2d');
  g.clearRect(0,0,w,h);const vals=p.s.map(v=>ok(v)?v:0);const mx=Math.max(2,...vals)*1.1;const X=k=>6+k*(w-12)/Math.max(1,N-1),Y=v=>h-10-v/mx*(h-24);
  g.strokeStyle='#DDE2E8';g.lineWidth=2;g.beginPath();g.moveTo(6,h-10);g.lineTo(w-6,h-10);g.stroke();
  g.setLineDash([6,6]);g.strokeStyle='#8CA0C3';for(const k of [C.snapIdx,C.relIdx]){if(k==null)continue;g.beginPath();g.moveTo(X(k),4);g.lineTo(X(k),h-10);g.stroke();}g.setLineDash([]);
  g.strokeStyle='#013369';g.lineWidth=4;g.beginPath();vals.forEach((v,k)=>{k?g.lineTo(X(k),Y(v)):g.moveTo(X(k),Y(v));});g.stroke();
  g.fillStyle='#D50A0A';g.beginPath();g.arc(X(cur),Y(vals[cur]),7,0,7);g.fill();
  g.fillStyle='#5B6770';g.font='20px Arial';g.fillText('vitesse (yd/s)  max '+mx.toFixed(1),8,18);}

function renderPanel(){const el=document.getElementById('panel');
  if(sel==null||!byId[sel]){el.innerHTML='<div class="empty"><b>Fiche joueur</b><br><br>Clique sur un joueur du terrain pour afficher son identité et ses statistiques.</div>';return;}
  const p=byId[sel],i=cur,col=p.side==='off'?C.offColor:C.defColor,tc=p.side==='off'?C.offText:C.defText,logo=p.side==='off'?C.offLogo:C.defLogo,tm=p.side==='off'?C.offName:C.defName;
  const n=nearestOpp(p,i),qb=C.qb!=null?byId[C.qb]:null,dq=qb&&qb.id!==p.id?dist(p,qb,i):null;
  const bx=D.ball.x[i],by=D.ball.y[i],db=(ok(bx)&&ok(p.x[i]))?Math.hypot(p.x[i]-bx,p.y[i]-by):null;
  const depth=(C.los!=null&&ok(p.x[i]))?p.x[i]-C.los:null;
  const isT=(C.target===p.id);
  const kmh=ok(p.s[i])?p.s[i]*YD2KMH:null;
  let h='<div class="ph" style="background:'+col+';color:'+(tcOf(col))+'">'+(logo?'<img src="'+logo+'" alt="'+p.team+'">':'')+
   '<div style="flex:1"><div class="nm">'+p.name+'</div><div class="sub">'+tm+' · '+p.pos+(p.lined?' (aligné '+p.lined+')':'')+'</div></div><div class="num">#'+(p.jersey!=null?p.jersey:'')+'</div></div>';
  h+='<div class="sec"><h4>Identité'+(isT?'<span class="tag">RECEVEUR VISÉ</span>':'')+'</h4>'+
   '<div class="kv"><span>Rôle sur l\'action</span><span>'+(p.role||'—')+'</span></div>'+
   '<div class="kv"><span>Taille / poids</span><span>'+(p.height||'—')+' · '+(p.weight?p.weight+' lb':'—')+'</span></div>'+
   '<div class="kv"><span>Université</span><span>'+(p.college||'—')+'</span></div></div>';
  h+='<div class="sec"><h4>À cet instant (t = '+D.t[i].toFixed(1)+' s)</h4><div class="big"><div><div class="v">'+f1(p.s[i])+'</div><div class="k">yd/s · '+(kmh!=null?kmh.toFixed(0):'—')+' km/h</div></div>'+
   '<div><div class="v">'+f1(p.a[i])+'</div><div class="k">accél. yd/s²</div></div></div>'+
   '<div class="kv"><span>Profondeur vs mêlée</span><span>'+(depth!=null?(depth>=0?'+':'')+depth.toFixed(1)+' yd':'—')+'</span></div>'+
   '<div class="kv"><span>'+(p.side==='off'?'Défenseur le plus proche':'Attaquant le plus proche')+'</span><span>'+(n?n.p.name+' · '+n.d.toFixed(1)+' yd':'—')+'</span></div>'+
   '<div class="kv"><span>Distance au QB</span><span>'+(dq!=null?dq.toFixed(1)+' yd':'—')+'</span></div>'+
   '<div class="kv"><span>Distance au ballon</span><span>'+(db!=null?db.toFixed(1)+' yd':'—')+'</span></div>'+
   '<div class="kv"><span>Direction / orientation</span><span>'+(ok(p.dir[i])?p.dir[i].toFixed(0)+'°':'—')+' / '+(ok(p.o[i])?p.o[i].toFixed(0)+'°':'—')+'</span></div></div>';
  h+='<div class="sec"><h4>Sur toute l\'action</h4>'+(p.pm?
   '<div class="kv"><span>Vitesse max</span><span>'+f1(p.pm.vmax)+' yd/s</span></div><div class="kv"><span>Accélération max</span><span>'+f1(p.pm.amax)+' yd/s²</span></div>'+
   '<div class="kv"><span>Distance parcourue (depuis le snap)</span><span>'+f1(p.pm.dist)+' yd</span></div><div class="kv"><span>Changements de direction</span><span>'+(p.pm.cuts!=null?p.pm.cuts:'—')+'</span></div>':'<div class="hint">Non disponible</div>')+
   '<canvas id="spark"></canvas></div>';
  h+='<div class="sec"><h4>Explosiveness Index (saison)</h4>'+(p.season?
   '<div class="big"><div><div class="v">'+p.season.EI.toFixed(0)+'</div><div class="k">indice /100</div></div><div><div class="v">'+p.season.rank+'<span style="font-size:15px;color:#5B6770">/'+p.season.n+'</span></div><div class="k">rang '+p.season.group+'</div></div></div>'+
   '<div class="kv"><span>V90 / A90</span><span>'+f1(p.season.vmax)+' yd/s · '+f1(p.season.amax)+' yd/s²</span></div><div class="kv"><span>Cuts par action</span><span>'+f2(p.season.cuts)+'</span></div>'+
   '<div class="kv"><span>Actions analysées</span><span>'+p.season.plays+'</span></div><div class="hint">Indice maison, non officiel NFL.</div>':'<div class="hint">Moins de 40 actions : joueur non classé.</div>')+'</div>';
  el.innerHTML=h;spark(p);}
function tcOf(c){const h=c.replace('#','');const r=parseInt(h.substr(0,2),16),g=parseInt(h.substr(2,2),16),b=parseInt(h.substr(4,2),16);return(0.2126*r+0.7152*g+0.0722*b)/255>0.45?'#111111':'#FFFFFF';}

// --- contrôles
function setCur(i){cur=Math.max(0,Math.min(N-1,i));updateScene();}
function stop(){if(timer){clearInterval(timer);timer=null;}document.getElementById('play').textContent='Lecture';}
function start(r){stop();rate=r;if(cur>=N-1)cur=0;document.getElementById('play').textContent='Pause';
  timer=setInterval(()=>{if(cur>=N-1){stop();return;}setCur(cur+1);},100/rate);}
document.getElementById('play').onclick=()=>{timer?stop():start(1);};
document.getElementById('slow').onclick=()=>{start(0.4);};
document.getElementById('gosnap').onclick=()=>{stop();if(C.snapIdx!=null)setCur(C.snapIdx);};
document.getElementById('gorel').onclick=()=>{stop();if(C.relIdx!=null)setCur(C.relIdx);};
const sl=document.getElementById('slider');sl.max=N-1;sl.oninput=()=>{stop();setCur(parseInt(sl.value));};
document.querySelectorAll('[data-cam]').forEach(b=>b.onclick=()=>{document.querySelectorAll('[data-cam]').forEach(x=>x.classList.remove('on'));b.classList.add('on');
  Plotly.relayout(gd,{'scene.camera':CAMS[b.dataset.cam]});});
function toggle(id,get,set,after){const b=document.getElementById(id);b.onclick=()=>{set(!get());b.classList.toggle('on',get());after&&after();updateScene();};}
toggle('tvec',()=>showVec,v=>showVec=v);toggle('ttrail',()=>showTrail,v=>showTrail=v);
toggle('tspace',()=>showSpace,v=>showSpace=v,()=>Plotly.restyle(gd,{visible:showSpace},[IDX.heat]));

C.offText=tcOf(C.offColor);C.defText=tcOf(C.defColor);
const conf={displaylogo:false,responsive:true,modeBarButtonsToRemove:['toImage','resetCameraLastSave3d','hoverClosest3d','orbitRotation','tableRotation']};
Plotly.newPlot(gd,FIG.data,FIG.layout,conf).then(()=>{
  gd.on('plotly_click',ev=>{const pt=ev&&ev.points&&ev.points[0];if(!pt)return;
    if(pt.curveNumber===IDX.pin_off)sel=off[pt.pointNumber].id;else if(pt.curveNumber===IDX.pin_def)sel=def[pt.pointNumber].id;else return;updateScene();});
  updateScene();});
window.addEventListener('resize',()=>{try{Plotly.Plots.resize(gd);}catch(e){}});
</script></body></html>"""
