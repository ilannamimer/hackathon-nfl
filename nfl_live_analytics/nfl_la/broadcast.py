"""Encarts « broadcast » en HTML/CSS (grands pourcentages, hiérarchie visuelle claire).

Chaque encart porte un badge de nature : MODÈLE (prédiction validée), CALCULÉ (formule), OBSERVÉ (donnée brute).
"""
from __future__ import annotations
from .config import C_ACCENT, C_BG, C_DEF, C_GOOD, C_MUTED, C_OFF, C_PANEL, C_TXT, C_WARN

CSS = f"""
<style>
.stApp {{ background: {C_BG}; color: {C_TXT}; }}
section[data-testid="stSidebar"] {{ background: #0E1729; }}
.bc-card {{ background: linear-gradient(135deg,{C_PANEL} 0%,#0F2038 100%); border-left: 6px solid {C_ACCENT};
  border-radius: 10px; padding: 16px 20px; margin: 6px 0 12px 0; box-shadow: 0 4px 18px rgba(0,0,0,.35); }}
.bc-title {{ font: 800 13px/1 'Arial Black',Arial; letter-spacing: .18em; color: {C_ACCENT}; text-transform: uppercase; }}
.bc-sub {{ color: {C_MUTED}; font-size: 12px; margin-top: 2px; }}
.bc-big {{ font: 900 60px/1.05 'Arial Black',Arial; color: #fff; }}
.bc-big small {{ font-size: 26px; color: {C_MUTED}; }}
.bc-row {{ display:flex; gap:18px; align-items:flex-end; flex-wrap:wrap; }}
.bc-kv {{ min-width: 120px; }}
.bc-k {{ color: {C_MUTED}; font-size: 11px; letter-spacing:.08em; text-transform: uppercase; }}
.bc-v {{ font: 800 26px/1.1 Arial; color:#fff; }}
.bc-badge {{ display:inline-block; font-size:10px; padding:2px 7px; border-radius:20px; margin-left:8px; letter-spacing:.06em;
  vertical-align: middle; font-weight:700; }}
.b-model {{ background:#1B3A6B; color:#9CC6FF; }} .b-calc {{ background:#2E3B2A; color:#B6F29B; }}
.b-obs {{ background:#44361A; color:#FFD98A; }}
.bc-bar {{ height: 10px; border-radius: 6px; background:#22304D; overflow:hidden; margin-top:8px; }}
.bc-bar > div {{ height:100%; border-radius:6px; }}
.bc-note {{ color:{C_MUTED}; font-size:11.5px; margin-top:10px; line-height:1.4; }}
</style>
"""
BADGE = {"model": '<span class="bc-badge b-model">MODÈLE</span>', "calc": '<span class="bc-badge b-calc">CALCULÉ</span>',
         "obs": '<span class="bc-badge b-obs">OBSERVÉ</span>'}


def _fmt(v, d=1, unit=""):
    return "—" if v is None or v != v else f"{v:.{d}f}{unit}"


def prob_card(title: str, sub: str, p: float | None, kind: str, note: str = "", color: str | None = None) -> str:
    color = color or (C_GOOD if (p or 0) >= .6 else C_WARN if (p or 0) >= .4 else C_DEF)
    big = "—" if p is None or p != p else f"{p*100:.0f}<small>%</small>"
    width = 0 if p is None or p != p else p * 100
    return (f'<div class="bc-card" style="border-left-color:{color}"><div class="bc-title">{title}{BADGE[kind]}</div>'
            f'<div class="bc-sub">{sub}</div><div class="bc-big">{big}</div>'
            f'<div class="bc-bar"><div style="width:{width}%;background:{color}"></div></div>'
            f'<div class="bc-note">{note}</div></div>')


def kv_card(title: str, sub: str, kvs: list[tuple[str, str]], kind: str, note: str = "", color: str = C_ACCENT) -> str:
    items = "".join(f'<div class="bc-kv"><div class="bc-k">{k}</div><div class="bc-v">{v}</div></div>' for k, v in kvs)
    return (f'<div class="bc-card" style="border-left-color:{color}"><div class="bc-title">{title}{BADGE[kind]}</div>'
            f'<div class="bc-sub">{sub}</div><div class="bc-row" style="margin-top:10px">{items}</div>'
            f'<div class="bc-note">{note}</div></div>')


def header(play_row) -> str:
    q, clock = int(play_row.quarter), play_row.gameClock
    dn = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(int(play_row.down), "—") if play_row.down == play_row.down else "—"
    return (f'<div class="bc-card" style="border-left-color:{C_OFF};padding:10px 18px"><div class="bc-row" style="align-items:center">'
            f'<div class="bc-kv"><div class="bc-k">{play_row.visitorTeamAbbr} @ {play_row.homeTeamAbbr}</div>'
            f'<div class="bc-v">{int(play_row.preSnapVisitorScore)} – {int(play_row.preSnapHomeScore)}</div></div>'
            f'<div class="bc-kv"><div class="bc-k">Quart-temps</div><div class="bc-v">Q{q} · {clock}</div></div>'
            f'<div class="bc-kv"><div class="bc-k">Situation</div><div class="bc-v">{dn} & {int(play_row.yardsToGo)}</div></div>'
            f'<div class="bc-kv"><div class="bc-k">Attaque</div><div class="bc-v" style="color:{C_OFF}">{play_row.possessionTeam}</div></div>'
            f'<div class="bc-kv"><div class="bc-k">Défense</div><div class="bc-v" style="color:{C_DEF}">{play_row.defensiveTeam}</div></div>'
            f'</div></div>')
