"""Habillage « NFL.com » : CSS global, bandeau supérieur, scorebug, encarts de stats.

Chaque encart porte un badge de nature : MODÈLE (prédiction validée), CALCULÉ (formule), OBSERVÉ (donnée brute).
Polices : Barlow / Barlow Condensed (Google Fonts), équivalents libres de la police propriétaire de la NFL.
"""
from __future__ import annotations
from . import teams
from .config import C_GOOD, C_MUTED, C_NAVY, C_RED, C_WARN

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700;800&family=Barlow:wght@400;500;600;700&display=swap');
html, body, .stApp, .stMarkdown, p, label, button, input, textarea, [data-testid="stSidebar"], [data-baseweb="select"],
[data-testid="stDataFrame"], [data-testid="stCaptionContainer"] {{ font-family: 'Barlow', Arial, sans-serif; }}
.stApp {{ background: #FFFFFF; color: #111827; }}
.block-container {{ padding-top: 0.6rem; max-width: 1500px; }}
[data-testid="stHeader"] {{ background: transparent; height: 0; }}
[data-testid="stSidebar"] {{ background: #F2F4F7; border-right: 1px solid #DDE2E8; }}
h1, h2, h3, h4 {{ font-family: 'Barlow Condensed', Arial Narrow, sans-serif !important; text-transform: uppercase;
  font-weight: 800 !important; letter-spacing: .02em; color: {C_NAVY}; }}
/* onglets façon navigation NFL */
.stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 3px solid {C_NAVY}; background:#fff; }}
.stTabs button[data-baseweb="tab"] {{ font-family:'Barlow Condensed',sans-serif; font-weight:700; font-size:1.08rem;
  text-transform:uppercase; letter-spacing:.05em; padding: 10px 18px; color:#333; }}
.stTabs button[data-baseweb="tab"][aria-selected="true"] {{ color:#fff; background:{C_NAVY}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color:{C_RED}; height:4px; }}
/* bandeau supérieur */
.nfl-top {{ background:{C_NAVY}; color:#fff; display:flex; align-items:center; gap:14px; padding:8px 18px; border-bottom:4px solid {C_RED};
  margin-bottom:10px; }}
.nfl-top img {{ height:38px; }}
.nfl-top .t1 {{ font:800 26px/1 'Barlow Condensed',sans-serif; letter-spacing:.08em; text-transform:uppercase; }}
.nfl-top .t2 {{ font:500 13px 'Barlow',sans-serif; color:#C9D6EA; margin-left:auto; letter-spacing:.04em; }}
/* scorebug */
.sb {{ display:flex; align-items:stretch; border:1px solid #DDE2E8; border-radius:4px; overflow:hidden; margin-bottom:8px;
  box-shadow:0 2px 8px rgba(1,51,105,.08); background:#fff; }}
.sb-team {{ flex:1; display:flex; align-items:center; gap:12px; padding:8px 16px; border-bottom:6px solid; }}
.sb-team.home {{ flex-direction:row-reverse; text-align:right; }}
.sb-team img {{ height:54px; width:54px; object-fit:contain; }}
.sb-abbr {{ font:800 34px/1 'Barlow Condensed',sans-serif; color:#111827; }}
.sb-name {{ font:500 12px 'Barlow',sans-serif; color:{C_MUTED}; text-transform:uppercase; letter-spacing:.06em; }}
.sb-score {{ font:800 46px/1 'Barlow Condensed',sans-serif; color:{C_NAVY}; margin:0 10px; }}
.sb-mid {{ background:{C_NAVY}; color:#fff; padding:6px 24px; display:flex; flex-direction:column; justify-content:center; align-items:center; min-width:230px; }}
.sb-mid .q {{ font:800 24px/1.1 'Barlow Condensed',sans-serif; letter-spacing:.06em; }}
.sb-mid .d {{ font:700 17px/1.2 'Barlow Condensed',sans-serif; color:#FFD60A; letter-spacing:.05em; text-transform:uppercase; }}
.sb-ball {{ color:{C_RED}; font-size:18px; margin:0 4px; }}
/* encarts */
.bc-card {{ background:#fff; border:1px solid #DDE2E8; border-left:8px solid {C_NAVY}; border-radius:4px; padding:14px 20px; margin:6px 0 12px 0;
  box-shadow:0 2px 8px rgba(1,51,105,.08); }}
.bc-title {{ font:800 17px/1 'Barlow Condensed',sans-serif; letter-spacing:.1em; color:{C_NAVY}; text-transform:uppercase; }}
.bc-sub {{ color:{C_MUTED}; font-size:13px; margin-top:3px; }}
.bc-big {{ font:800 70px/1.0 'Barlow Condensed',sans-serif; color:#111827; margin-top:4px; }}
.bc-big small {{ font-size:30px; color:{C_MUTED}; }}
.bc-row {{ display:flex; gap:22px; align-items:flex-end; flex-wrap:wrap; }}
.bc-kv {{ min-width:110px; }}
.bc-k {{ color:{C_MUTED}; font:600 12px 'Barlow',sans-serif; letter-spacing:.08em; text-transform:uppercase; }}
.bc-v {{ font:800 32px/1.1 'Barlow Condensed',sans-serif; color:#111827; }}
.bc-badge {{ display:inline-block; font:700 10px 'Barlow',sans-serif; padding:2px 8px; border-radius:2px; margin-left:10px;
  vertical-align:middle; letter-spacing:.08em; }}
.b-model {{ background:#E3ECF8; color:{C_NAVY}; }} .b-calc {{ background:#E2F3E8; color:#14663A; }} .b-obs {{ background:#FFF0D1; color:#8A5A00; }}
.bc-bar {{ height:10px; border-radius:2px; background:#E6EAF0; overflow:hidden; margin-top:10px; }}
.bc-bar > div {{ height:100%; }}
.bc-note {{ color:{C_MUTED}; font-size:12.5px; margin-top:10px; line-height:1.4; }}
</style>
"""
BADGE = {"model": '<span class="bc-badge b-model">MODÈLE</span>', "calc": '<span class="bc-badge b-calc">CALCULÉ</span>',
         "obs": '<span class="bc-badge b-obs">OBSERVÉ</span>'}


def top_bar() -> str:
    return (f'<div class="nfl-top"><img src="{teams.NFL_LOGO}" alt="NFL"><span class="t1">Live Analytics</span>'
            f'<span class="t2">Replay analytique · saison 2021 · tracking Big Data Bowl</span></div>')


def logo_img(abbr: str, h: int = 54) -> str:
    u = teams.logo_url(abbr)
    return f'<img src="{u}" alt="{abbr}" style="height:{h}px;width:{h}px;object-fit:contain">' if u else ""


def prob_card(title: str, sub: str, p: float | None, kind: str, note: str = "", color: str | None = None) -> str:
    color = color or (C_GOOD if (p or 0) >= .6 else C_WARN if (p or 0) >= .4 else C_RED)
    big = "—" if p is None or p != p else f"{p*100:.0f}<small>%</small>"
    width = 0 if p is None or p != p else p * 100
    return (f'<div class="bc-card" style="border-left-color:{color}"><div class="bc-title">{title}{BADGE[kind]}</div>'
            f'<div class="bc-sub">{sub}</div><div class="bc-big">{big}</div>'
            f'<div class="bc-bar"><div style="width:{width}%;background:{color}"></div></div>'
            f'<div class="bc-note">{note}</div></div>')


def kv_card(title: str, sub: str, kvs: list[tuple[str, str]], kind: str, note: str = "", color: str = C_NAVY) -> str:
    items = "".join(f'<div class="bc-kv"><div class="bc-k">{k}</div><div class="bc-v">{v}</div></div>' for k, v in kvs)
    return (f'<div class="bc-card" style="border-left-color:{color}"><div class="bc-title">{title}{BADGE[kind]}</div>'
            f'<div class="bc-sub">{sub}</div><div class="bc-row" style="margin-top:10px">{items}</div>'
            f'<div class="bc-note">{note}</div></div>')


def header(r) -> str:
    """Scorebug : visiteur | quart-temps / situation | domicile, avec logos et couleurs d'équipe."""
    dn = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(int(r.down), "") if r.down == r.down else ""
    sit = f"{dn} &amp; {int(r.yardsToGo)}" if dn else "—"
    ball = '<span class="sb-ball">●</span>'
    def side(abbr, score, cls):
        poss = ball if abbr == r.possessionTeam else ""
        return (f'<div class="sb-team {cls}" style="border-bottom-color:{teams.color(abbr)}">{logo_img(abbr)}'
                f'<div><div class="sb-abbr">{abbr}{poss}</div><div class="sb-name">{teams.name(abbr)}</div></div>'
                f'<div style="flex:1"></div><div class="sb-score">{int(score)}</div></div>')
    return ('<div class="sb">' + side(r.visitorTeamAbbr, r.preSnapVisitorScore, "away")
            + f'<div class="sb-mid"><div class="q">Q{int(r.quarter)} · {r.gameClock}</div><div class="d">{sit} · {r.possessionTeam} ball</div></div>'
            + side(r.homeTeamAbbr, r.preSnapHomeScore, "home") + '</div>')