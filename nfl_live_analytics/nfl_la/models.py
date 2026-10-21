"""Modèles probabilistes (régression logistique + gradient boosting) évalués SANS fuite de données.

Protocole : validation croisée groupée par match (GroupKFold sur gameId) — aucune action d'un match
de test n'est vue à l'entraînement. Les probabilités affichées dans l'app sont les prédictions
« out-of-fold » : chaque action est prédite par un modèle qui n'a jamais vu son match.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NUM_FEATURES = [
    "down", "yardsToGo", "yards_to_goal", "quarter", "score_diff",
    "time_to_throw", "qb_speed", "qb_depth", "rush_min_dist", "rush_mean2_dist", "n_rush",
    "n_rush_within5", "pressure_at_release", "max_pressure", "n_block", "n_routes",
    "best_sep", "mean_sep", "n_open3", "target_sep", "target_dist", "target_depth", "target_speed",
    "defendersInBox", "pff_playAction",
]
CAT_FEATURES = ["offenseFormation", "pff_passCoverageType", "dropBackType"]
FEATURES = NUM_FEATURES + CAT_FEATURES

LABELS = {
    "p_complete": "Probabilité de complétion",
    "p_explosive": "Probabilité de passe explosive (≥ 15 yds)",
}


def build_matrix(df: pd.DataFrame, cats: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Retourne X (catégories encodées en entiers, NaN conservés) et le dictionnaire des catégories."""
    X = df[FEATURES].copy()
    cats = cats or {c: sorted(df[c].dropna().astype(str).unique()) for c in CAT_FEATURES}
    for c in CAT_FEATURES:
        X[c] = pd.Categorical(X[c].astype(object).where(X[c].notna(), None).astype("object"),
                              categories=cats[c]).codes.astype(float)
        X.loc[X[c] < 0, c] = np.nan
    return X.astype(float), cats


def _hgb(cols: list[str] | None = None) -> HistGradientBoostingClassifier:
    cat_mask = [c in CAT_FEATURES for c in (cols or FEATURES)]
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=180, l2_regularization=2.0,
                                          min_samples_leaf=40, categorical_features=cat_mask, random_state=0)


def _logit() -> object:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.3, max_iter=2000))


def expected_calibration_error(y, p, bins=10) -> float:
    q = pd.qcut(p, bins, duplicates="drop")
    d = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q", observed=True).agg(y=("y", "mean"), p=("p", "mean"),
                                                                              n=("y", "size"))
    return float((d.n * (d.y - d.p).abs()).sum() / d.n.sum())


def calibration_table(y, p, bins=10) -> pd.DataFrame:
    q = pd.qcut(p, bins, duplicates="drop")
    d = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q", observed=True).agg(
        observed=("y", "mean"), predicted=("p", "mean"), n=("y", "size")).reset_index(drop=True)
    return d


def scores(y, p) -> dict:
    base = np.full(len(y), np.mean(y))
    b, b0 = brier_score_loss(y, p), brier_score_loss(y, base)
    return {"brier": float(b), "brier_baseline": float(b0), "brier_skill": float(1 - b / b0),
            "auc": float(roc_auc_score(y, p)), "log_loss": float(log_loss(y, p)),
            "ece": expected_calibration_error(y, p), "n": int(len(y)), "prevalence": float(np.mean(y))}


def cross_validate(df: pd.DataFrame, y: pd.Series, n_splits: int = 5, drop: list[str] | None = None) -> dict:
    """OOF de la régression logistique et du boosting ; renvoie prédictions + scores.

    `drop` : colonnes à retirer (ablation, ex. variables de la cible) pour mesurer leur contribution."""
    X, cats = build_matrix(df)
    cols = [c for c in FEATURES if c not in (drop or [])]
    X = X[cols]
    groups = df["gameId"].to_numpy()
    oof = {"logit": np.zeros(len(df)), "hgb": np.zeros(len(df))}
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        assert not set(groups[tr]) & set(groups[te]), "fuite : un match est dans train ET test"
        for name, mk in (("logit", _logit), ("hgb", lambda: _hgb(cols))):
            m = mk().fit(X.iloc[tr], y.iloc[tr])
            oof[name][te] = m.predict_proba(X.iloc[te])[:, 1]
    return {"oof": oof, "scores": {k: scores(y.to_numpy(), v) for k, v in oof.items()}, "cats": cats}


def fit_final(df: pd.DataFrame, y: pd.Series, kind: str):
    """Modèle final entraîné sur toutes les actions (sert uniquement au simulateur « what-if »)."""
    X, cats = build_matrix(df)
    m = (_hgb if kind == "hgb" else _logit)().fit(X, y)
    return m, cats
