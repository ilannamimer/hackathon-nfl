"""Pipeline complet : extraction des features -> entraînement/évaluation -> fichiers lus par l'app.

Usage :  python build_features.py [--force]
Sorties (dossier outputs/) :
    play_features.parquet  une ligne par action + probabilités out-of-fold
    player_index.parquet   classement PLAYER EXPLOSIVENESS INDEX
    report.json            métriques d'évaluation, validations, limites chiffrées
    models.joblib          modèles finaux (simulateur what-if)
"""
from __future__ import annotations
import argparse, json, re, sys, time
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from nfl_la import data, metrics, models
from nfl_la.config import OUT_DIR, find_data_dir

POS_GROUP = {"QB": "QB", "RB": "RB/FB", "FB": "RB/FB", "WR": "WR", "TE": "TE", "T": "OL", "G": "OL", "C": "OL",
             "DE": "DL", "DT": "DL", "NT": "DL", "OLB": "LB", "ILB": "LB", "MLB": "LB", "LB": "LB",
             "CB": "DB", "FS": "DB", "SS": "DB", "DB": "DB"}
EI_WEIGHTS = {"vmax": 0.40, "amax": 0.35, "cuts": 0.25}
MIN_PLAYS = 40


# ------------------------------------------------------------------ 1. extraction
def extract(T: dict, dd) -> tuple[pd.DataFrame, pd.DataFrame]:
    plays, pff = T["plays"], T["pffScoutingData"]
    pff_g = {k: v for k, v in pff.groupby("gameId")}
    play_idx = plays.set_index(["gameId", "playId"], drop=False)
    feats, pp = [], []
    games = T["games"].gameId.tolist()
    t0 = time.time()
    for i, gid in enumerate(games, 1):
        try:
            tr = data.load_tracking_game(dd, gid)
        except data.DataError as e:
            print("  !", e, file=sys.stderr); continue
        roles_g = {k: v for k, v in pff_g.get(gid, pff.iloc[:0]).groupby("playId")}
        for pid, g in tr.groupby("playId"):
            try:
                feats.append(metrics.play_features(g, roles_g.get(pid, pff.iloc[:0]), play_idx.loc[(gid, pid)]))
            except Exception as e:  # une action défaillante ne doit pas bloquer le pipeline
                print(f"  ! action {gid}/{pid} ignorée : {e}", file=sys.stderr)
        pm = metrics.player_play_metrics(tr); pm["gameId"] = gid; pp.append(pm)
        print(f"[{i:3d}/{len(games)}] {gid}  {time.time()-t0:5.0f}s", flush=True)
    return pd.DataFrame(feats), pd.concat(pp, ignore_index=True)


# ------------------------------------------------------------------ 2. cible réelle (texte de l'action)
def _norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

def abbrev(display: str) -> str:
    parts = [p for p in display.replace(".", ". ").split() if p.lower() not in ("jr.", "sr.", "ii", "iii", "iv")]
    return _norm(parts[0][0] + "".join(parts[1:])) if len(parts) > 1 else _norm(display)

def targets_from_description(feat: pd.DataFrame, plays: pd.DataFrame, pff: pd.DataFrame,
                             players: pd.DataFrame) -> pd.Series:
    """Cible réelle = receveur (rôle 'Pass Route') dont le nom abrégé apparaît dans la description de la passe.

    N'utilise AUCUNE information sur le résultat (complétion, gain) : uniquement l'identité du receveur visé.
    """
    name = players.set_index("nflId")["displayName"].to_dict()
    routes = pff[pff.pff_role == "Pass Route"].groupby(["gameId", "playId"])["nflId"].apply(list).to_dict()
    desc = plays.set_index(["gameId", "playId"])["playDescription"].to_dict()
    out = {}
    for gid, pid in zip(feat.gameId, feat.playId):
        text = desc.get((gid, pid), "")
        m = re.search(r"(?:pass|intended for|to)\s.*", text) and re.split(r"\bto\b|intended for", text)
        tail = _norm(" ".join(m[1:])) if m and len(m) > 1 else ""
        hit = [r for r in routes.get((gid, pid), []) if r in name and abbrev(name[r]) and abbrev(name[r]) in tail]
        out[(gid, pid)] = hit[0] if len(hit) == 1 else np.nan
    return pd.Series(out)


def main(force: bool = False) -> None:
    dd = find_data_dir()
    if dd is None:
        sys.exit("Données introuvables : clonez le dépôt officiel ou définissez NFL_DATA_DIR.")
    T = data.load_tables(dd)
    raw_f, raw_p = OUT_DIR / "_features_raw.parquet", OUT_DIR / "_player_play.parquet"
    if force or not raw_f.exists():
        feat, pp = extract(T, dd)
        feat.to_parquet(raw_f); pp.to_parquet(raw_p)
    feat, pp = pd.read_parquet(raw_f), pd.read_parquet(raw_p)
    plays, games, pff, players = T["plays"], T["games"], T["pffScoutingData"], T["players"]
    report: dict = {"n_games": int(games.gameId.nunique()), "n_plays_dataset": int(len(plays)),
                    "season": int(games.season.iloc[0]), "weeks": sorted(games.week.unique().tolist())}

    # -- table action : plays + games + features
    df = plays.merge(games[["gameId", "week", "homeTeamAbbr", "visitorTeamAbbr", "label"]], on="gameId")
    df = df.merge(feat.drop(columns=[c for c in feat.columns if c in df.columns and c not in ("gameId", "playId")]),
                  on=["gameId", "playId"], how="left")
    home = df.possessionTeam == df.homeTeamAbbr
    df["score_diff"] = np.where(home, df.preSnapHomeScore - df.preSnapVisitorScore,
                                df.preSnapVisitorScore - df.preSnapHomeScore)
    df["is_attempt"] = df.passResult.isin(["C", "I", "IN"]) & df.release_frame.notna()
    df["complete"] = (df.passResult == "C").astype(int)
    df["explosive"] = (df.prePenaltyPlayResult >= 15).astype(int)

    # -- validation de l'heuristique de cible
    tgt_true = targets_from_description(df[df.is_attempt][["gameId", "playId"]], plays, pff, players)
    df = df.merge(tgt_true.rename("target_true").reset_index().rename(
        columns={"level_0": "gameId", "level_1": "playId"}), on=["gameId", "playId"], how="left")
    ok = df.target_true.notna() & df.target_id.notna()
    report["target_heuristic"] = {
        "n_parsed": int(ok.sum()), "n_attempts": int(df.is_attempt.sum()),
        "agreement": float((df.loc[ok, "target_true"] == df.loc[ok, "target_id"]).mean())}
    print("Cible heuristique vs texte :", report["target_heuristic"])
    # cible retenue : la vraie (texte) si identifiée, sinon heuristique
    df["target_final"] = df.target_true.fillna(df.target_id)
    df["target_source"] = np.where(df.target_true.notna(), "texte", np.where(df.target_id.notna(), "heuristique", ""))

    # -- recalcul des features de cible avec la cible retenue (là où elle diffère)
    diff = df[df.is_attempt & df.target_true.notna() & (df.target_true != df.target_id)]
    print(f"Recalcul des features de cible pour {len(diff)} actions…")
    for gid, sub in diff.groupby("gameId"):
        tr = data.load_tracking_game(dd, gid)
        for _, r in sub.iterrows():
            g = tr[tr.playId == r.playId]
            pv, rel = metrics.pivots(g), int(r.release_frame)
            team = g[g.nflId.notna()].drop_duplicates("nflId").set_index("nflId")["team"].to_dict()
            dfn = [i for i in pv["x"].columns if team.get(i) == r.defensiveTeam]
            X, Y, S = pv["x"].loc[rel], pv["y"].loc[rel], pv["s"].loc[rel]
            t = r.target_true
            if t in X.index:
                i = r.name
                df.at[i, "target_sep"] = float(metrics.min_dist(np.array([X[t]]), np.array([Y[t]]),
                                               X[dfn].to_numpy(), Y[dfn].to_numpy())[0])
                qb = next(iter(pff[(pff.gameId == gid) & (pff.playId == r.playId) & (pff.pff_role == "Pass")].nflId), None)
                if qb in X.index:
                    df.at[i, "target_dist"] = float(np.hypot(X[t] - X[qb], Y[t] - Y[qb]))
                df.at[i, "target_depth"] = float(X[t] - r.los_x)
                df.at[i, "target_speed"] = float(S[t])

    # -- 3. modèles
    A = df[df.is_attempt].reset_index(drop=True)
    final, results = {}, {}
    for target, col in (("p_complete", "complete"), ("p_explosive", "explosive")):
        res = models.cross_validate(A, A[col])
        best = min(res["scores"], key=lambda k: res["scores"][k]["brier"])
        results[target] = {"scores": res["scores"], "best": best,
                           "calibration": models.calibration_table(A[col].to_numpy(), res["oof"][best]).to_dict("list")}
        A[target] = res["oof"][best]
        m, cats = models.fit_final(A, A[col], best)
        final[target] = {"model": m, "cats": cats}
        print(target, best, json.dumps(res["scores"][best], indent=0))
    report["models"] = results
    # ablation : performance SANS les variables de la cible (receveur visé) -> contribution honnête de cette information
    tfeat = [c for c in models.FEATURES if c.startswith("target_")]
    report["ablation_no_target"] = {t: {k: v for k, v in models.cross_validate(A, A[c], drop=tfeat)["scores"]["hgb"].items()
                                        if k in ("brier", "brier_skill", "auc")} for t, c in (("p_complete", "complete"), ("p_explosive", "explosive"))}
    print("Ablation sans cible :", report["ablation_no_target"])
    df = df.merge(A[["gameId", "playId", "p_complete", "p_explosive"]], on=["gameId", "playId"], how="left")
    medians = A[models.NUM_FEATURES].median().to_dict()
    joblib.dump({"models": final, "medians": medians, "features": models.FEATURES}, OUT_DIR / "models.joblib")

    # -- 4. pression : validité de l'indice vs résultats PFF
    flags = pff.groupby(["gameId", "playId"]).agg(
        hit=("pff_hit", "max"), hurry=("pff_hurry", "max"), sack=("pff_sack", "max")).fillna(0).reset_index()
    df = df.merge(flags, on=["gameId", "playId"], how="left")
    df["pressured"] = ((df.hit.fillna(0) + df.hurry.fillna(0) + df.sack.fillna(0)) > 0).astype(int)
    d = df[df.max_pressure.notna()]
    tiers = pd.cut(d.max_pressure, [-1, 25, 50, 75, 101], labels=["0-25", "25-50", "50-75", "75-100"])
    tab = d.groupby(tiers, observed=True).agg(n=("pressured", "size"), pressure_rate=("pressured", "mean"),
                                              sack_rate=("sack", "mean")).reset_index()
    report["pressure"] = {"auc_vs_pressured": float(roc_auc_score(d.pressured, d.max_pressure)),
                          "auc_vs_sack": float(roc_auc_score(d.sack.fillna(0), d.max_pressure)),
                          "tiers": tab.rename(columns={"max_pressure": "tier"}).astype({"tier": str}).to_dict("list")}

    # -- 5. 4th down (descriptif seulement : uniquement les dropbacks sont dans les données)
    f4 = df[df.down == 4].copy()
    f4["converted"] = (f4.prePenaltyPlayResult >= f4.yardsToGo) & f4.passResult.isin(["C", "R"])
    f4["bucket"] = pd.cut(f4.yardsToGo, [0, 2, 4, 7, 99], labels=["1-2", "3-4", "5-7", "8+"])
    t4 = f4.groupby("bucket", observed=True).agg(n=("converted", "size"), rate=("converted", "mean")).reset_index()
    report["fourth_down"] = {"n": int(len(f4)), "table": t4.astype({"bucket": str}).to_dict("list")}

    # -- 6. PLAYER EXPLOSIVENESS INDEX
    ei = pp.merge(players[["nflId", "displayName", "officialPosition"]], on="nflId", how="left")
    ei["group"] = ei.officialPosition.map(POS_GROUP)
    g = ei.groupby("nflId").agg(player=("displayName", "first"), position=("officialPosition", "first"),
                                group=("group", "first"), plays=("playId", "size"),
                                vmax=("vmax", lambda s: s.quantile(0.9)), amax=("amax", lambda s: s.quantile(0.9)),
                                cuts=("cuts", "mean"), top_speed=("vmax", "max")).reset_index()
    g = g[(g.plays >= MIN_PLAYS) & g.group.notna()].copy()
    for c in EI_WEIGHTS:
        g[f"pct_{c}"] = g.groupby("group")[c].rank(pct=True)
    g["EI"] = 100 * sum(w * g[f"pct_{c}"] for c, w in EI_WEIGHTS.items())
    alts = [(1/3, 1/3, 1/3), (0.5, 0.3, 0.2), (0.3, 0.3, 0.4), (0.5, 0.5, 0.0)]
    report["explosiveness"] = {
        "weights": EI_WEIGHTS, "min_plays": MIN_PLAYS, "n_players": int(len(g)),
        "sensitivity_spearman": [float(spearmanr(g.EI, 100 * (a[0]*g.pct_vmax + a[1]*g.pct_amax + a[2]*g.pct_cuts)).statistic)
                                 for a in alts],
        "sensitivity_weights": alts,
        "corr_vmax_amax": float(spearmanr(g.vmax, g.amax).statistic),
        "corr_vmax_cuts": float(spearmanr(g.vmax, g.cuts).statistic)}
    g.to_parquet(OUT_DIR / "player_index.parquet")

    keep = [c for c in df.columns if not c.startswith("foul")]
    df[keep].to_parquet(OUT_DIR / "play_features.parquet")
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print("Terminé ✔  ->", OUT_DIR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
