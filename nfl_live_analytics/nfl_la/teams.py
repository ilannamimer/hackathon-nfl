"""Identité visuelle des équipes : noms, couleurs (primaire/secondaire) et logos.

Les logos sont des marques déposées : ils ne sont PAS inclus dans le projet. L'app les charge à l'affichage
depuis le CDN public d'ESPN (connexion internet requise). Usage pédagogique/démo uniquement.
"""
from __future__ import annotations

TEAMS: dict[str, tuple[str, str, str]] = {   # abbr -> (nom, primaire, secondaire)
    "ARI": ("Arizona Cardinals", "#97233F", "#FFB612"), "ATL": ("Atlanta Falcons", "#A71930", "#000000"),
    "BAL": ("Baltimore Ravens", "#241773", "#9E7C0C"), "BUF": ("Buffalo Bills", "#00338D", "#C60C30"),
    "CAR": ("Carolina Panthers", "#0085CA", "#101820"), "CHI": ("Chicago Bears", "#0B162A", "#C83803"),
    "CIN": ("Cincinnati Bengals", "#FB4F14", "#000000"), "CLE": ("Cleveland Browns", "#311D00", "#FF3C00"),
    "DAL": ("Dallas Cowboys", "#003594", "#869397"), "DEN": ("Denver Broncos", "#FB4F14", "#002244"),
    "DET": ("Detroit Lions", "#0076B6", "#B0B7BC"), "GB": ("Green Bay Packers", "#203731", "#FFB612"),
    "HOU": ("Houston Texans", "#03202F", "#A71930"), "IND": ("Indianapolis Colts", "#002C5F", "#A2AAAD"),
    "JAX": ("Jacksonville Jaguars", "#006778", "#D7A22A"), "KC": ("Kansas City Chiefs", "#E31837", "#FFB81C"),
    "LV": ("Las Vegas Raiders", "#000000", "#A5ACAF"), "LAC": ("Los Angeles Chargers", "#0080C6", "#FFC20E"),
    "LA": ("Los Angeles Rams", "#003594", "#FFA300"), "MIA": ("Miami Dolphins", "#008E97", "#FC4C02"),
    "MIN": ("Minnesota Vikings", "#4F2683", "#FFC62F"), "NE": ("New England Patriots", "#002244", "#C60C30"),
    "NO": ("New Orleans Saints", "#D3BC8D", "#101820"), "NYG": ("New York Giants", "#0B2265", "#A71930"),
    "NYJ": ("New York Jets", "#125740", "#000000"), "PHI": ("Philadelphia Eagles", "#004C54", "#A5ACAF"),
    "PIT": ("Pittsburgh Steelers", "#FFB612", "#101820"), "SF": ("San Francisco 49ers", "#AA0000", "#B3995D"),
    "SEA": ("Seattle Seahawks", "#002244", "#69BE28"), "TB": ("Tampa Bay Buccaneers", "#D50A0A", "#34302B"),
    "TEN": ("Tennessee Titans", "#0C2340", "#4B92DB"), "WAS": ("Washington Commanders", "#5A1414", "#FFB612"),
}
_ESPN = {"WAS": "wsh", "LA": "lar"}          # codes ESPN qui diffèrent de l'abréviation des données
NFL_LOGO = "https://a.espncdn.com/i/teamlogos/leagues/500/nfl.png"


def logo_url(abbr: str) -> str | None:
    return f"https://a.espncdn.com/i/teamlogos/nfl/500/{_ESPN.get(abbr, abbr.lower())}.png" if abbr in TEAMS else None


def name(abbr: str) -> str:
    return TEAMS.get(abbr, (abbr,))[0]


def color(abbr: str, secondary: bool = False, default: str = "#013369") -> str:
    t = TEAMS.get(abbr)
    return default if t is None else t[2 if secondary else 1]


def _rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#"); return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def luminance(h: str) -> float:
    r, g, b = (v / 255 for v in _rgb(h)); return 0.2126 * r + 0.7152 * g + 0.0722 * b


def text_on(h: str) -> str:
    """Couleur de texte lisible sur le fond `h`."""
    return "#111111" if luminance(h) > 0.45 else "#FFFFFF"


def matchup_colors(off: str, de: str) -> tuple[str, str]:
    """Couleurs attaque/défense ; si trop proches, la défense passe à sa couleur secondaire."""
    a, d = color(off), color(de)
    dist = sum(abs(x - y) for x, y in zip(_rgb(a), _rgb(d)))
    if dist < 150:
        d = color(de, secondary=True)
        if sum(abs(x - y) for x, y in zip(_rgb(a), _rgb(d))) < 150:
            d = "#FFFFFF" if luminance(a) < 0.45 else "#000000"
    return a, d
