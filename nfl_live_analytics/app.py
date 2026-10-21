"""NFL LIVE ANALYTICS — dashboard Streamlit (prototype).

Lancer :  streamlit run app.py
Prérequis : `python build_features.py` exécuté une fois (génère outputs/).
IMPORTANT : l'outil rejoue des actions HISTORIQUES (saison 2021, semaines 1-8). Il n'y a aucune connexion live.
"""
from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components

from nfl_la import broadcast as bc, data, field, metrics, models, teams, viewer3d
from nfl_la.config import (FONT, C_NAVY, C_RED, C_TRAIL, C_ACCENT, C_BALL, C_BG, C_DEF, C_GOOD, C_MUTED, C_OFF, C_PANEL, C_TXT, C_WARN, FPS, OUT_DIR,
                           find_data_dir)

st.set_page_config(page_title="NFL LIVE ANALYTICS", page_icon=teams.NFL_LOGO, layout="wide")
st.markdown(bc.CSS, unsafe_allow_html=True)
st.markdown(bc.top_bar(), unsafe_allow_html=True)
# police Barlow par défaut dans tous les graphiques Plotly
pio.templates["nfl"] = go.layout.Template(layout=dict(font=dict(family=FONT)))
pio.templates.default = "plotly+nfl"

# ============================================================== chargement (avec gestion d'erreurs)
DD = find_data_dir()
if DD is None:
    st.error("Données introuvables. Clonez `nfl-big-data-bowl-regional-event-data` dans le dossier du projet "
             "ou définissez la variable d'environnement `NFL_DATA_DIR`.")
    st.stop()
need = [OUT_DIR / f for f in ("play_features.parquet", "player_index.parquet", "report.json", "models.joblib")]
if not all(p.exists() for p in need):
    st.warning("Fichiers d'analyse manquants. Lancez d'abord : `python build_features.py` (≈ 5 min).")
    st.stop()


@st.cache_data(show_spinner=False)
def load_all(dd):
    T = data.load_tables(dd)
    return (T, pd.read_parquet(OUT_DIR / "play_features.parquet"), pd.read_parquet(OUT_DIR / "player_index.parquet"),
            json.loads((OUT_DIR / "report.json").read_text()))


@st.cache_resource(show_spinner=False)
def load_models():
    return joblib.load(OUT_DIR / "models.joblib")


@st.cache_data(show_spinner=False, max_entries=3)
def load_game(dd, gid):
    return data.load_tracking_game(dd, gid)


try:
    T, PF, PI, REP = load_all(DD)
    MOD = load_models()
except data.DataError as e:
    st.error(str(e)); st.stop()
PLAYERS = T["players"].set_index("nflId")

# ============================================================== barre latérale : sélection
st.sidebar.markdown("### Sélection")
st.sidebar.caption("Actions historiques · saison 2021 (S1-S8) · pas de données live")
games = T["games"].sort_values("gameId")
glabel = st.sidebar.selectbox("Match", games.gameId, format_func=lambda g: games.set_index("gameId").label[g])
_g = games.set_index("gameId").loc[glabel]
st.sidebar.markdown(f'<div style="display:flex;align-items:center;justify-content:center;gap:14px;margin:4px 0 10px 0">'
                    f'{bc.logo_img(_g.visitorTeamAbbr, 44)}<span style="font:800 20px \'Barlow Condensed\';color:{C_NAVY}">@</span>'
                    f'{bc.logo_img(_g.homeTeamAbbr, 44)}</div>', unsafe_allow_html=True)
sub = PF[PF.gameId == glabel].sort_values("playId")
only_att = st.sidebar.checkbox("Passes lancées seulement", True)
if only_att:
    sub = sub[sub.is_attempt]
if sub.empty:
    st.warning("Aucune action pour ce filtre."); st.stop()
def plabel(r):
    d = f"{int(r.down)}&{int(r.yardsToGo)}" if r.down == r.down and r.down > 0 else "—"
    return f"Q{int(r.quarter)} {r.gameClock} · {d} · {r.playDescription[:70]}"
sub = sub.assign(_l=sub.apply(plabel, axis=1)).reset_index(drop=True)
pid = st.sidebar.selectbox("Action", sub.playId, format_func=lambda p: sub.set_index("playId")._l[p])
P = sub[sub.playId == pid].iloc[0]

try:
    G_all = load_game(DD, int(glabel))
except data.DataError as e:
    st.error(str(e)); st.stop()
g = G_all[G_all.playId == pid]
roles = T["pffScoutingData"].query("gameId == @glabel and playId == @pid")
role = roles.set_index("nflId")["pff_role"].to_dict(); lined = roles.set_index("nflId")["pff_positionLinedUp"].to_dict()
team = g[g.nflId.notna()].drop_duplicates("nflId").set_index("nflId")["team"].to_dict()
ids = list(team)
meta = pd.DataFrame({"name": [PLAYERS.displayName.get(i, f"#{i:.0f}") for i in ids], "pos": [lined.get(i, "?") for i in ids]}, index=ids)
qb = next((i for i in ids if role.get(i) == "Pass"), None)
routes = [i for i in ids if role.get(i) == "Pass Route"]
rushers = [i for i in ids if role.get(i) == "Pass Rush"]
defense = [i for i in ids if team.get(i) == P.defensiveTeam]
snap, rel = metrics.event_frame(g, metrics.SNAP_EVENTS), metrics.event_frame(g, metrics.RELEASE_EVENTS)
los = float(P.los_x) if P.los_x == P.los_x else None

default_rcv = P.target_final if P.target_final == P.target_final and P.target_final in routes else (routes[0] if routes else None)
rcv = st.sidebar.selectbox("Receveur analysé", routes, index=routes.index(default_rcv) if default_rcv in routes else 0,
                           format_func=lambda i: f"{meta.name[i]} ({meta.pos[i]})") if routes else None
if P.target_source:
    st.sidebar.caption(f"Receveur visé : **{meta.name.get(P.target_final, '?')}** (source : {P.target_source})")

st.markdown(bc.header(P), unsafe_allow_html=True)
st.caption(P.playDescription)
tabs = st.tabs(["Broadcast", "Terrain 3D", "Analyse de l'action", "Joueurs", "Probability Center", "Méthodes & limites"])

# ============================================================== calculs communs
def sep_for(rid):
    return metrics.separation_timeline(g, rid, defense) if rid is not None and defense else None

sep_tl = sep_for(rcv)
press_tl = metrics.qb_pressure_timeline(g, qb, rushers) if qb is not None and rushers else pd.DataFrame()
def at(df, col, f):
    if df is None or df.empty or f is None: return None
    r = df[df.frameId == f]
    return float(r[col].iloc[0]) if len(r) else None

# ============================================================== 1. BROADCAST
with tabs[0]:
    cA, cB = st.columns([1, 1])
    with cA:
        if P.is_attempt:
            st.markdown(bc.prob_card("PASS COMPLETION", "Probabilité de complétion au moment du lancer", P.p_complete, "model",
                        f"Modèle {REP['models']['p_complete']['best']} · Brier {REP['models']['p_complete']['scores'][REP['models']['p_complete']['best']]['brier']:.3f} "
                        f"(référence {REP['models']['p_complete']['scores'][REP['models']['p_complete']['best']]['brier_baseline']:.3f}). "
                        "Prédiction out-of-fold : ce match n'a pas servi à entraîner ce modèle."), unsafe_allow_html=True)
            st.markdown(bc.prob_card("EXPLOSIVE PASS", "Probabilité d'un gain ≥ 15 yards", P.p_explosive, "model",
                        "Même protocole. Résultat réel de l'action : "
                        f"<b>{int(P.prePenaltyPlayResult)} yds</b> · passe {'complétée' if P.complete else 'non complétée'}.",
                        color=C_NAVY), unsafe_allow_html=True)
        else:
            st.info("Cette action n'est pas une passe lancée (sack / course du QB) : pas de probabilité de passe.")
        if P.down == 4:
            t4 = pd.DataFrame(REP["fourth_down"]["table"])
            st.markdown(bc.kv_card("4TH DOWN", f"4ème & {int(P.yardsToGo)} · {REP['fourth_down']['n']} dropbacks observés sur 4ème tentative", [
                (f"{r.bucket} yds", f"{r.rate*100:.0f}%<span style='font-size:12px;color:{C_MUTED}'> (n={r.n})</span>") for r in t4.itertuples()],
                "obs", "Attention : taux de conversion OBSERVÉS des passes uniquement. Punts, field goals et courses sont absents des données : "
                "aucune recommandation de décision n'est possible.", color=C_WARN), unsafe_allow_html=True)
    with cB:
        if sep_tl is not None and rel is not None:
            s_rel, s_max = at(sep_tl, "sep", rel), float(sep_tl[sep_tl.frameId <= rel].sep.max())
            v_rel = at(sep_tl, "speed", rel)
            near = defense[0] if False else sep_tl.loc[sep_tl.frameId == rel, "nearest_def"].iloc[0]
            st.markdown(bc.kv_card("RECEIVER SEPARATION", f"{meta.name[rcv]} ({meta.pos[rcv]}) · défenseur le plus proche : {meta.name.get(near, '?')}",
                        [("Séparation au lancer", f"{s_rel:.1f} yd"), ("Séparation max", f"{s_max:.1f} yd"),
                         ("Vitesse au lancer", f"{v_rel:.1f} yd/s")], "calc",
                        "Distance euclidienne au défenseur adverse le plus proche (tracking à 10 Hz).", color=C_OFF), unsafe_allow_html=True)
        if not press_tl.empty:
            p_rel = at(press_tl, "pressure", rel) if rel else float(press_tl.pressure.iloc[-1])
            d_rel = at(press_tl, "d_rush", rel) if rel else float(press_tl.d_rush.iloc[-1])
            tiers = pd.DataFrame(REP["pressure"]["tiers"])
            lvl = 0 if p_rel < 25 else 1 if p_rel < 50 else 2 if p_rel < 75 else 3
            obs = tiers.pressure_rate.iloc[lvl] if lvl < len(tiers) else None
            st.markdown(bc.kv_card("QB PRESSURE", "Plus proche rusher du QB", [
                ("Pressure Index", f"{p_rel:.0f}/100"), ("Distance", f"{d_rel:.1f} yd"), ("Max sur l'action", f"{press_tl.pressure.max():.0f}/100"),
                ("Pression PFF observée (même palier)", f"{obs*100:.0f}%" if obs is not None else "—")], "calc",
                f"Indice calculé (formule dans « Méthodes »), PAS une probabilité. Validité : AUC {REP['pressure']['auc_vs_pressured']:.2f} "
                "vs hit/hurry/sack PFF. La dernière valeur est un taux observé par palier.", color=C_DEF), unsafe_allow_html=True)
    # graphiques
    f1, f2 = st.columns(2)
    if sep_tl is not None:
        fg = go.Figure(go.Scatter(x=(sep_tl.frameId - sep_tl.frameId.min()) / FPS, y=sep_tl.sep, mode="lines", line=dict(color=C_OFF, width=3), fill="tozeroy",
                                  fillcolor="rgba(47,155,255,.15)", name="Séparation"))
        for f, lab in ((snap, "snap"), (rel, "lancer")):
            if f: fg.add_vline(x=(f - sep_tl.frameId.min()) / FPS, line_dash="dash", line_color=C_MUTED, annotation_text=lab, annotation_font_color=C_MUTED)
        fg.update_layout(title="Évolution de la séparation (yd)", height=280, paper_bgcolor=C_BG, plot_bgcolor=C_PANEL, font=dict(color=C_TXT),
                         margin=dict(l=10, r=10, t=40, b=10), xaxis_title="secondes", yaxis_title="yd")
        f1.plotly_chart(fg, theme=None, width="stretch")
    if not press_tl.empty:
        fg = go.Figure(go.Scatter(x=(press_tl.frameId - press_tl.frameId.min()) / FPS, y=press_tl.pressure, mode="lines", line=dict(color=C_DEF, width=3),
                                  fill="tozeroy", fillcolor="rgba(255,77,94,.15)"))
        fg.update_layout(title="Pressure Index (0-100)", height=280, paper_bgcolor=C_BG, plot_bgcolor=C_PANEL, font=dict(color=C_TXT),
                         margin=dict(l=10, r=10, t=40, b=10), xaxis_title="secondes", yaxis=dict(range=[0, 100]))
        f2.plotly_chart(fg, theme=None, width="stretch")

# ============================================================== 2. TERRAIN 3D
with tabs[1]:
    html3d = viewer3d.build_html(g, meta, role, lined, PLAYERS, PI, P, rcv, qb, los,
                                 float(P.yardsToGo) if P.yardsToGo == P.yardsToGo else None, snap, rel, height=760)
    if hasattr(st, "iframe"):          # Streamlit récent ; components.html est déprécié
        st.iframe(html3d, height=790)
    else:
        components.html(html3d, height=790, scrolling=False)
    st.caption(f"Attaque : {teams.name(P.possessionTeam)} (end zone gauche) · défense : {teams.name(P.defensiveTeam)} (end zone droite). "
               "Ligne bleue = mêlée, ligne jaune = 1er down, vecteurs = déplacement projeté à +0,8 s, losange = ballon.")

# ============================================================== 3. ANALYSE
with tabs[2]:
    a1, a2 = st.columns([3, 2])
    with a1:
        st.subheader("Joueurs au moment du lancer" if rel else "Joueurs (fin du suivi)")
        f = rel or int(g.frameId.max())
        snapdf = g[(g.frameId == f) & g.nflId.notna()].copy()
        snapdf["Joueur"] = snapdf.nflId.map(meta.name); snapdf["Poste"] = snapdf.nflId.map(meta.pos)
        snapdf["Camp"] = np.where(snapdf.team == P.possessionTeam, "Attaque", "Défense")
        vmax = g[g.nflId.notna() & (g.frameId >= (snap or 1))].groupby("nflId").s.max()
        snapdf["Vitesse max (yd/s)"] = snapdf.nflId.map(vmax).round(2)
        st.dataframe(snapdf[["Joueur", "Poste", "Camp", "s", "a", "Vitesse max (yd/s)"]].rename(columns={"s": "Vitesse (yd/s)", "a": "Accél. (yd/s²)"})
                     .sort_values("Vitesse (yd/s)", ascending=False), hide_index=True, width="stretch", height=360)
    with a2:
        st.subheader("Défenseurs les plus proches du QB")
        if qb is not None:
            q = g[(g.frameId == f) & (g.nflId == qb)].iloc[0]
            dd_ = g[(g.frameId == f) & g.nflId.isin(defense)].copy()
            dd_["d"] = np.hypot(dd_.x - q.x, dd_.y - q.y)
            dd_["Joueur"] = dd_.nflId.map(meta.name); dd_["Poste"] = dd_.nflId.map(meta.pos)
            st.dataframe(dd_.nsmallest(4, "d")[["Joueur", "Poste", "d", "s"]].rename(columns={"d": "Distance (yd)", "s": "Vitesse (yd/s)"}).round(2),
                         hide_index=True, width="stretch")
            st.caption("Le porteur de balle identifiable est le QB (dropback). Les données ne contiennent pas de porteur après la passe.")
        if los is not None and rel:
            share = field.space_share(g[g.frameId == rel], P.possessionTeam, los)
            if share is not None:
                st.markdown(bc.kv_card("OFFENSIVE SPACE", "Zone LOS → LOS+20 yds au lancer", [("Contrôlée par l'attaque", f"{share*100:.0f}%")], "calc",
                            "Chaque cellule va à l'équipe qui l'atteindrait en premier (positions projetées à +0,5 s)."), unsafe_allow_html=True)
    st.subheader("Situations à surveiller (règles explicites)")
    flags = []
    if P.get("time_to_throw") == P.get("time_to_throw") and P.time_to_throw >= 3.2: flags.append(f"Lancer tardif : {P.time_to_throw:.1f}s après le snap.")
    if P.get("pressure_at_release") == P.get("pressure_at_release") and P.pressure_at_release >= 60: flags.append(f"QB sous forte pression au lancer (index {P.pressure_at_release:.0f}).")
    if P.get("target_sep") == P.get("target_sep") and P.target_sep >= 4: flags.append(f"Receveur visé très ouvert ({P.target_sep:.1f} yd).")
    if P.get("target_sep") == P.get("target_sep") and P.target_sep <= 1.2: flags.append(f"Passe dans une fenêtre serrée ({P.target_sep:.1f} yd du défenseur).")
    if P.get("n_open3") == P.get("n_open3") and P.n_open3 == 0: flags.append("Aucun receveur à plus de 3 yds d'un défenseur au lancer.")
    if P.is_attempt and P.p_complete == P.p_complete and (P.p_complete >= .6) != bool(P.complete) and abs(P.p_complete - .5) > .15:
        flags.append("Résultat à contre-courant du modèle (probabilité " + f"{P.p_complete*100:.0f}% → {'complétée' if P.complete else 'non complétée'}).")
    for fl in flags or ["Rien d'atypique selon les règles ci-dessus."]: st.markdown(f"- {fl}")
    st.info("Non disponible dans les données : receveur ciblé officiel (estimé à partir du texte de l'action), événements de plaquage (3 sur 8 557 actions), "
            "actions de course/punt/field goal, résultat final des matchs.")

# ============================================================== 4. JOUEURS
with tabs[3]:
    st.subheader("PLAYER EXPLOSIVENESS INDEX (indice maison — non officiel NFL)")
    E = REP["explosiveness"]
    st.caption(f"P = rang percentile du joueur dans son groupe de postes (OL, DL, LB, DB, WR, TE, RB/FB, QB). Minimum {E['min_plays']} actions. "
               "vmax et amax = 90ᵉ percentile par action ; cuts = nombre moyen de changements de direction ≥ 45° en 0,3 s.")
    grp = st.multiselect("Groupes de postes", sorted(PI.group.unique()), default=["WR", "DB", "RB/FB"])
    mp = st.slider("Actions minimum", E["min_plays"], int(PI.plays.max()), E["min_plays"])
    V = PI[PI.group.isin(grp) & (PI.plays >= mp)].sort_values("EI", ascending=False)
    c1, c2 = st.columns([2, 3])
    c1.dataframe(V[["player", "position", "plays", "EI", "vmax", "amax", "cuts", "top_speed"]].round(2).rename(columns={
        "player": "Joueur", "position": "Poste", "plays": "Actions", "vmax": "V90 (yd/s)", "amax": "A90 (yd/s²)", "cuts": "Cuts/action", "top_speed": "V max"}),
        hide_index=True, width="stretch", height=470)
    fg = go.Figure(go.Scatter(x=V.vmax, y=V.amax, mode="markers", text=V.player + " (" + V.position + ")", hoverinfo="text",
                              marker=dict(size=8 + 14 * V.cuts / max(V.cuts.max(), 1e-6), color=V.EI, colorscale="Turbo", showscale=True,
                                          colorbar=dict(title="EI"), line=dict(color="white", width=.5))))
    fg.update_layout(title="Vitesse vs accélération (taille = changements de direction)", height=470, paper_bgcolor=C_BG, plot_bgcolor=C_PANEL,
                     font=dict(color=C_TXT), xaxis_title="V90 (yd/s)", yaxis_title="A90 (yd/s²)", margin=dict(l=10, r=10, t=40, b=10))
    c2.plotly_chart(fg, theme=None, width="stretch")
    with st.expander("Justification des pondérations & robustesse"):
        st.markdown(f"""
- **Vitesse (0,40)** : mesure la plus fiable (directement issue du tracking).
- **Accélération (0,35)** : explosivité pure, mais dérivée donc plus bruitée → poids légèrement inférieur.
- **Changements de direction (0,25)** : agilité ; compte plus rare et dépendant du rôle (les receveurs en font plus que les linemen) → poids le plus faible.
- Les pondérations sont des **choix a priori**, pas des valeurs estimées. Robustesse : corrélation de Spearman de l'indice avec 4 pondérations alternatives
  = {', '.join(f'{x:.2f}' for x in E['sensitivity_spearman'])} (classement stable si proche de 1).
- Corrélations entre composantes : vmax↔amax ρ={E['corr_vmax_amax']:.2f}, vmax↔cuts ρ={E['corr_vmax_cuts']:.2f}.
- Limite : uniquement des actions de passe (dropbacks) ; l'indice mesure l'explosivité *observée dans ce contexte*.""")

# ============================================================== 5. PROBABILITY CENTER
with tabs[4]:
    st.subheader("Qualité des modèles (validation croisée groupée par match)")
    rows = []
    for k, lab in models.LABELS.items():
        for name, s in REP["models"][k]["scores"].items():
            rows.append({"Cible": lab, "Modèle": {"logit": "Régression logistique", "hgb": "Gradient boosting"}[name] + (" (retenu)" if name == REP["models"][k]["best"] else ""),
                         "Brier": s["brier"], "Brier référence": s["brier_baseline"], "Skill": s["brier_skill"], "AUC": s["auc"], "ECE": s["ece"], "n": s["n"]})
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
    st.caption("Brier : plus bas = mieux. Skill = 1 − Brier/Brier(taux moyen constant) ; > 0 = le modèle apporte de l'information. ECE = erreur de calibration moyenne. (retenu) = modèle retenu.")
    ab = REP["ablation_no_target"]
    st.caption(f"**Honnêteté sur la performance** : le receveur visé (identifié via le texte de l'action, connu au moment du lancer) apporte l'essentiel du signal. "
               f"Sans les variables de la cible, la complétion tombe à AUC {ab['p_complete']['auc']:.2f} (skill {ab['p_complete']['brier_skill']:.2f}) "
               f"et la passe explosive à AUC {ab['p_explosive']['auc']:.2f} (skill {ab['p_explosive']['brier_skill']:.2f}).")
    cc = st.columns(2)
    for col, (k, lab) in zip(cc, models.LABELS.items()):
        cal = pd.DataFrame(REP["models"][k]["calibration"])
        fg = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=C_MUTED, dash="dash"), name="calibration parfaite"),
                        go.Scatter(x=cal.predicted, y=cal.observed, mode="lines+markers", line=dict(color=C_ACCENT, width=3), name="modèle")])
        fg.update_layout(title=f"Calibration — {lab}", height=330, paper_bgcolor=C_BG, plot_bgcolor=C_PANEL, font=dict(color=C_TXT),
                         xaxis_title="probabilité prédite", yaxis_title="fréquence observée", margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
        col.plotly_chart(fg, theme=None, width="stretch")
    st.subheader("Simulateur « what-if » (scénario fictif — pas une action réelle)")
    med = MOD["medians"]; s1, s2, s3, s4 = st.columns(4)
    v = {"time_to_throw": s1.slider("Temps de lancer (s)", 1.0, 5.0, float(round(med["time_to_throw"], 1)), .1),
         "target_sep": s2.slider("Séparation du receveur (yd)", 0.0, 10.0, float(round(med["target_sep"], 1)), .1),
         "pressure_at_release": s3.slider("Pressure Index au lancer", 0, 100, int(med["pressure_at_release"])),
         "target_depth": s4.slider("Profondeur de la cible (yd)", -5.0, 40.0, float(round(med["target_depth"], 1)), .5)}
    row = {c: med[c] for c in models.NUM_FEATURES}; row.update(v)
    row["rush_min_dist"] = float(np.interp(v["pressure_at_release"], [0, 100], [8, 0.5]))   # cohérence pression ↔ distance
    row["target_dist"] = max(1.0, v["target_depth"] + 4)
    X1 = pd.DataFrame([{**row, **{c: None for c in models.CAT_FEATURES}}])
    out = {}
    for k in ("p_complete", "p_explosive"):
        Xm, _ = models.build_matrix(X1, MOD["models"][k]["cats"]); out[k] = float(MOD["models"][k]["model"].predict_proba(Xm)[0, 1])
    w1, w2 = st.columns(2)
    w1.markdown(bc.prob_card("PASS COMPLETION", "Scénario fictif", out["p_complete"], "model", "Autres variables = médiane du jeu de données."), unsafe_allow_html=True)
    w2.markdown(bc.prob_card("EXPLOSIVE PASS", "Scénario fictif", out["p_explosive"], "model", "Modèle final entraîné sur toutes les actions (usage illustratif).", color=C_NAVY), unsafe_allow_html=True)
    st.subheader("Probabilités NON calculables avec ces données")
    st.markdown("""
| Probabilité demandée | Pourquoi elle n'est pas calculée | Solution réaliste |
|---|---|---|
| **Victoire des équipes** | Aucun résultat final de match dans le dépôt, seulement les scores avant action sur des passes | Joindre le play-by-play *nflverse* 2021 (`old_game_id` = `gameId`) |
| **Réussite d'un field goal** | Aucune tentative de field goal (dropbacks uniquement) | nflverse : `field_goal_attempt`, distance, résultat |
| **Décision 4ᵉ tentative (go / punt / FG)** | 231 passes seulement, pas de punts ni FG → biais de sélection | nflverse (modèle de décision) ; ici : taux observés uniquement |
| **Plaquage** | 3 événements « tackle » sur 8 557 actions | Dataset BDB avec événements de plaquage |
""")

# ============================================================== 6. MÉTHODES
with tabs[5]:
    th = REP["target_heuristic"]
    st.markdown(f"""
### Données réellement utilisées
Dépôt officiel : **{REP['n_games']} matchs**, saison **{REP['season']}**, semaines {REP['weeks'][0]}–{REP['weeks'][-1]}, **{REP['n_plays_dataset']:,} actions** — uniquement des *dropbacks* (passes, sacks, scrambles).
Tracking à 10 Hz (x, y, s, a, dis, o, dir, event) + scouting PFF (rôles, postes alignés, pression). Coordonnées normalisées : l'attaque va toujours vers la droite.

### Nature de chaque valeur
- **OBSERVÉ** : position, vitesse, accélération, résultat de la passe, gain, score avant action.
- **CALCULÉ** : séparation, Pressure Index, contrôle de l'espace, Player Explosiveness Index (formules explicites).
- **MODÈLE** : probabilité de complétion, probabilité de passe explosive (prédictions out-of-fold).
- **FICTIF** : uniquement le simulateur « what-if », clairement étiqueté.

### Modèles
Features à l'instant du lancer : situation (down, distance, zone, score), temps de lancer, pression (distance/rapprochement des rushers), séparations des receveurs,
caractéristiques de la cible (séparation, profondeur, distance au QB), formation, couverture PFF. **Aucune information sur le résultat n'est utilisée.**
Validation : GroupKFold ×5 **par match** ; comparaison régression logistique vs gradient boosting ; métriques Brier, skill, AUC, ECE, courbes de calibration.

### Receveur ciblé
Absent des données. Identifié par le nom abrégé présent dans le texte de l'action ({th['n_parsed']:,} actions sur {th['n_attempts']:,}) ; sinon estimé par heuristique de trajectoire du ballon.
Accord heuristique ↔ texte : **{th['agreement']*100:.0f}%**.

### Pressure Index
`PI = 100 × (0,7 × clip((6 − d)/6, 0, 1) + 0,3 × clip(v_rapprochement/6, 0, 1))` avec d = distance au rusher le plus proche (yd) et v en yd/s sur 0,3 s.
Ce n'est **pas une probabilité** : AUC {REP['pressure']['auc_vs_pressured']:.2f} contre « hit/hurry/sack » PFF, {REP['pressure']['auc_vs_sack']:.2f} contre les sacks ; les taux par palier sont observés.

### Limites
- Passes seulement ; pas de courses, punts, field goals → pas de modèle de décision 4ᵉ tentative ni de win probability.
- 8 semaines de la saison 2021 (≈ 7 500 passes) : les modèles sont modestes, la complétion reste largement imprévisible avant le lancer.
- Pas de position du ballon à l'arrivée ni de receveur ciblé officiel ; pas de données live.
- Pondérations de l'Explosiveness Index et du Pressure Index = choix a priori (sensibilité fournie).
""")