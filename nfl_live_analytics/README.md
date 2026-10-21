# 🏈 NFL LIVE ANALYTICS

## README (version à rendre — 5 phrases)
**NFL LIVE ANALYTICS** est un dashboard Streamlit qui rejoue les actions de passe des 122 matchs du dépôt (saison 2021, semaines 1-8) avec un terrain animé, des encarts « broadcast » (probabilité de complétion, passe explosive, séparation du receveur, pression sur le QB) et un indice maison, le *Player Explosiveness Index*.
Les deux probabilités viennent de modèles entraînés puis validés par validation croisée groupée par match : la complétion atteint un Brier de 0,187 (référence 0,238) et une AUC de 0,77, mais l'AUC tombe à 0,69 sans l'information sur le receveur visé, donc l'essentiel du signal vient de la profondeur et de la séparation de la cible plutôt que de la situation de jeu.
Le Pressure Index (calculé, pas une probabilité) sépare nettement les situations : dans son palier le plus élevé, 96 % des actions ont une pression PFF et 48 % finissent en sack, contre 1 % de pression dans le palier le plus bas.
Les données ne contiennent que des dropbacks : aucune win probability, probabilité de field goal ou décision de 4ᵉ tentative n'est donc affichée, seulement des taux observés clairement étiquetés.
L'outil s'adresse à un commentateur qui cherche un encart visuel, à un analyste qui compare les receveurs et la pression, ou à un recruteur qui repère les profils explosifs.

## Installation et lancement
```bash
git clone https://github.com/ThompsonJamesBliss/nfl-big-data-bowl-regional-event-data.git   # dans ce dossier
pip install -r requirements.txt
python build_features.py        # ≈ 5 min, une seule fois (déjà exécuté : outputs/ est fourni)
streamlit run app.py
```
Si les données sont ailleurs : `NFL_DATA_DIR=/chemin/vers/.../data streamlit run app.py`.

## Structure
```
app.py                 interface Streamlit (6 onglets)
build_features.py      extraction -> modèles -> outputs/
nfl_la/data.py         chargement robuste, normalisation (attaque → droite)
nfl_la/metrics.py      séparation, pression, cible estimée, explosivité
nfl_la/models.py       logistique + gradient boosting, GroupKFold, Brier, calibration
nfl_la/field.py        terrain Plotly + animation + contrôle de l'espace
nfl_la/broadcast.py    encarts HTML/CSS
outputs/               play_features.parquet, player_index.parquet, report.json, models.joblib
```

## Ce que les données contiennent vraiment (audit)
| Fonctionnalité demandée | Données nécessaires | Disponible ? | Faisabilité |
|---|---|---|---|
| Probabilité de complétion | tracking + résultat | oui (`passResult`, tracking au lancer) | ✅ modèle validé |
| Gain important (≥ 15 yds) | idem + gain | oui (`prePenaltyPlayResult`) | ✅ modèle validé |
| Séparation receveur/défenseur | positions + rôles | oui (rôles PFF) | ✅ calculé |
| Pression sur le QB | positions + rôles + hit/hurry/sack | oui | ✅ indice + validation |
| Explosiveness Index | vitesse, accélération, direction | oui | ✅ calculé |
| Animation / terrain | tracking 10 Hz | oui | ✅ |
| Probabilité de plaquage | événements de plaquage | 3 sur 8 557 actions | ❌ remplacé par QB PRESSURE |
| Field goal | tentatives de FG | aucune (dropbacks seulement) | ❌ (nflverse requis) |
| 4ᵉ tentative go/punt/FG | toutes les 4ᵉ tentatives | 231 passes, pas de punts/FG | ⚠️ taux observés seulement |
| Win probability | résultat final des matchs | absent | ❌ (nflverse requis) |

⚠️ Le dépôt contient la saison **2021** (semaines 1-8), pas 2023 ; uniquement des **dropbacks** (8 557 actions).

## Statistiques et formules
- **Complétion / passe explosive** (MODÈLE) : gradient boosting vs régression logistique, features au lancer (situation, temps de lancer, pression, séparations, cible : séparation/profondeur/distance, formation, couverture). Prédictions affichées = *out-of-fold*.
- **Receveur visé** : lu dans le texte de l'action (6 793 actions), sinon estimé par la direction initiale du ballon (accord heuristique ↔ texte : 86 %).
- **Séparation** (CALCULÉ) : distance au défenseur adverse le plus proche.
- **Pressure Index** (CALCULÉ) : `100 × (0,7·clip((6−d)/6,0,1) + 0,3·clip(v_rapprochement/6,0,1))`. Pas une probabilité. La validation (AUC 0,90 vs pression PFF) est en partie circulaire, la pression PFF étant elle-même liée à la proximité des rushers ; les taux par palier sont observés.
- **Player Explosiveness Index** (CALCULÉ, non officiel) : `EI = 100 × (0,40·P_vmax + 0,35·P_amax + 0,25·P_cuts)`, P = rang percentile dans le groupe de postes, ≥ 40 actions. Pondérations a priori ; Spearman avec 4 pondérations alternatives : 0,95–0,99.
- **Contrôle de l'espace** (CALCULÉ) : cellule attribuée à l'équipe qui l'atteint en premier (positions projetées à +0,5 s).

## Limites
Passes uniquement ; 8 semaines ; modèles modestes (complétion peu prévisible sans connaître la cible) ; pas de données live (replay historique) ; `pff_passCoverage` est une annotation PFF faite après coup ; pondérations des indices non estimées.

## Capture d'écran idéale pour la présentation
Onglet **📺 Broadcast** sur une passe profonde, par exemple Prescott → A. Cooper (DAL, Semaine 1, 28 yds) : les cartes PASS COMPLETION et EXPLOSIVE PASS en haut à gauche, RECEIVER SEPARATION et QB PRESSURE à droite, les courbes de séparation et de pression en dessous. Puis une seconde capture de l'onglet **🏟 Terrain animé** avec « Contrôle de l'espace » activé, au moment du lancer.
