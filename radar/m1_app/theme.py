from __future__ import annotations

import streamlit as st

PAPER = "#F5F1E8"
CARD = "#FFFDF8"
INK = "#1E2321"
MUTED = "#6B6A60"
RULE = "#DDD6C6"
EMBER = "#B4531A"
BRICK = "#9E2B25"
BURNT = "#D2711F"
OCHRE = "#C9961A"
MOSS = "#5F7F4B"
PINE = "#2E5E4E"
STONE = "#9A978A"

RISK_COLORS = {
    "Critical": BRICK,
    "High": BURNT,
    "Medium": OCHRE,
    "Low": MOSS,
    "None identified": STONE,
    "Not assessed": STONE,
}

TIER_COLORS = {
    "Critical": BRICK,
    "High": BURNT,
    "Medium": OCHRE,
    "Low": MOSS,
    "Not assessed": STONE,
}

USAGE_COLORS = {
    "Yes": PINE,
    "Inconclusive": OCHRE,
    "No": STONE,
}

STRENGTH_COLORS = {
    "Confirmed": PINE,
    "Strong": MOSS,
    "Probable": OCHRE,
    "Marketing-only": STONE,
    "None": STONE,
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root{--paper:#F5F1E8;--card:#FFFDF8;--ink:#1E2321;--muted:#6B6A60;--rule:#DDD6C6;--ember:#B4531A;--pine:#2E5E4E;}
*, *:before, *:after{box-shadow:none !important;text-shadow:none !important;}
*:focus-visible{outline:2px solid var(--ember) !important;outline-offset:1px !important;}
html,body,[data-testid="stApp"],[data-testid="stAppViewContainer"]{background:var(--paper);color:var(--ink);font-family:'IBM Plex Sans',system-ui,-apple-system,sans-serif;}
.stAppHeader,header,[data-testid="stHeader"],[data-testid="stHeader"] *,[data-testid="stToolbar"],.stAppDeployButton,[data-testid="stAppDeployButton"],button[data-testid="stBaseButton-header"],#MainMenu,footer{display:none !important;visibility:hidden !important;height:0 !important;width:0 !important;overflow:hidden !important;}
.block-container{max-width:1180px;padding-top:2.2rem;padding-bottom:4rem;}
.mono{font-family:'IBM Plex Mono',ui-monospace,monospace;}
.hero{border-bottom:2px solid var(--ink);padding-bottom:1rem;margin-bottom:1rem;}
.eyebrow{font:500 .72rem 'IBM Plex Mono',monospace;letter-spacing:.16em;text-transform:uppercase;color:var(--ember);}
.hero-title{font-family:'Fraunces',Georgia,serif;font-weight:600;font-size:2.5rem;line-height:1.1;letter-spacing:-.015em;margin:.25rem 0 .4rem;color:var(--ink);}
.hero-sub{color:var(--muted);max-width:62ch;font-size:.98rem;}
.notice{border-left:3px solid var(--ember);background:#EFE6D2;padding:.7rem .95rem;font-size:.85rem;line-height:1.5;color:#4A463B;margin:.9rem 0 1.2rem;border-radius:2px;}
.headline{font-family:'Fraunces',Georgia,serif;font-weight:500;font-size:1.4rem;line-height:1.38;max-width:64ch;margin:.4rem 0 1.2rem;color:var(--ink);}
.grid{display:grid;gap:.9rem;margin-bottom:.9rem;}
.g4{grid-template-columns:repeat(4,minmax(0,1fr));}
.g2{grid-template-columns:repeat(2,minmax(0,1fr));}
@media(max-width:860px){.g4{grid-template-columns:repeat(2,minmax(0,1fr));}.g2{grid-template-columns:1fr;}}
@media(max-width:520px){.g4{grid-template-columns:1fr;}}
.kpi{background:var(--card);border:1px solid var(--rule);border-top:4px solid var(--c);padding:.85rem 1rem .9rem;border-radius:2px;}
.kpi-label{font:500 .7rem 'IBM Plex Mono',monospace;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);}
.kpi-value{font-family:'Fraunces',Georgia,serif;font-weight:600;font-size:2.4rem;line-height:1.1;margin:.3rem 0 .15rem;color:var(--ink);}
.kpi-note{font-size:.8rem;color:var(--muted);}
.card{background:var(--card);border:1px solid var(--rule);padding:1rem 1.1rem;border-radius:2px;}
.card-title{font:500 .72rem 'IBM Plex Mono',monospace;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin-bottom:.8rem;}
.stack{display:flex;height:22px;gap:2px;margin-bottom:.8rem;}
.seg{min-width:6px;}
.legend{list-style:none;margin:0;padding:0;display:flex;flex-wrap:wrap;gap:.35rem 1.1rem;font-size:.84rem;}
.legend li{display:flex;align-items:center;gap:.4rem;}
.legend i{width:10px;height:10px;display:inline-block;}
.legend b{font-family:'IBM Plex Mono',monospace;font-weight:500;}
.pill{display:inline-block;font-size:.72rem;font-weight:600;padding:.12rem .5rem;border-radius:2px;color:var(--c);background:color-mix(in srgb,var(--c) 12%,white);border:1px solid color-mix(in srgb,var(--c) 38%,white);white-space:nowrap;}
.meter{position:relative;display:flex;align-items:center;gap:.5rem;min-width:120px;}
.meter-track{position:relative;flex:1;height:6px;background:#E8E1D0;border-radius:2px;}
.meter-fill{height:100%;border-radius:2px;}
.meter-tick{position:absolute;top:-3px;width:1px;height:12px;background:var(--ink);opacity:.55;}
.meter span{font:500 .78rem 'IBM Plex Mono',monospace;width:2.8rem;text-align:right;}
.matrix{width:100%;border-collapse:collapse;font-size:.84rem;}
.matrix th{font:500 .68rem 'IBM Plex Mono',monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);text-align:center;padding:.3rem .4rem;}
.matrix th:first-child{text-align:left;}
.matrix td{border:2px solid var(--card);text-align:center;height:58px;vertical-align:middle;}
.matrix td.rowh{text-align:left;border-color:transparent;width:30%;}
.cell-n{font-family:'Fraunces',Georgia,serif;font-weight:600;font-size:1.25rem;line-height:1;}
.cell-ids{font:400 .64rem 'IBM Plex Mono',monospace;color:#4A463B;margin-top:.2rem;}
.cell-empty{color:#C4BBA6;}
.bars{display:flex;flex-direction:column;gap:.55rem;}
.bar-row{display:grid;grid-template-columns:38% 1fr 2rem;align-items:center;gap:.6rem;font-size:.84rem;}
.bar-track{height:10px;background:#E8E1D0;border-radius:2px;}
.bar-fill{height:100%;background:var(--ember);border-radius:2px;}
.bar-n{font-family:'IBM Plex Mono',monospace;text-align:right;}
.empty{color:var(--muted);font-size:.86rem;}
.section{font-family:'Fraunces',Georgia,serif;font-weight:600;font-size:1.2rem;margin:1.4rem 0 .7rem;}
.prio{background:var(--card);border:1px solid var(--rule);border-left:5px solid var(--c);padding:.7rem 1rem;margin-bottom:.55rem;border-radius:2px;}
.prio-head{display:flex;flex-wrap:wrap;align-items:center;gap:.5rem;}
.prio-head strong{font-size:.98rem;}
.prio-id{font:500 .74rem 'IBM Plex Mono',monospace;color:var(--muted);}
.prio p{margin:.4rem 0 0;font-size:.86rem;line-height:1.5;color:#3B382F;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;}
.tbl-wrap{overflow-x:auto;border:1px solid var(--rule);background:var(--card);border-radius:2px;}
.tbl{width:100%;border-collapse:collapse;font-size:.86rem;}
.tbl th{font:500 .68rem 'IBM Plex Mono',monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);text-align:left;padding:.65rem .8rem;border-bottom:2px solid var(--ink);white-space:nowrap;}
.tbl td{padding:.6rem .8rem;border-bottom:1px solid var(--rule);vertical-align:middle;}
.tbl tr:last-child td{border-bottom:0;}
.tbl tr:hover td{background:#F8F3E6;}
.tbl .num{font-family:'IBM Plex Mono',monospace;text-align:right;}
.vname{font-weight:600;}
.vid{display:block;font:400 .72rem 'IBM Plex Mono',monospace;color:var(--muted);}
.detail-label{font:500 .68rem 'IBM Plex Mono',monospace;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin:.8rem 0 .25rem;}
.detail-text{font-size:.9rem;line-height:1.55;color:#2B2A24;}
.chips{display:flex;flex-wrap:wrap;gap:.35rem;}
.chip{font-size:.78rem;padding:.12rem .55rem;border:1px solid var(--rule);background:#F1EBDB;border-radius:2px;}
.detail-list{margin:0;padding-left:1.1rem;font-size:.88rem;line-height:1.55;}
.stTabs [data-baseweb="tab-list"]{gap:1.6rem;border-bottom:1px solid var(--rule);background:transparent;}
.stTabs [data-baseweb="tab"]{font-family:'IBM Plex Sans',sans-serif;font-weight:500;color:var(--muted);padding-left:0;padding-right:0;background:transparent;border:none;}
.stTabs [aria-selected="true"]{color:var(--ink) !important;}
.stTabs [data-baseweb="tab-highlight"]{background:var(--ember) !important;height:3px;}
[data-testid="stExpander"]{background:var(--card) !important;border:1px solid var(--rule) !important;border-radius:2px !important;}
[data-testid="stFileUploaderDropzone"]{background:var(--card) !important;border:1px dashed #B9B09A !important;border-radius:2px !important;}
.stButton>button,.stDownloadButton>button{border-radius:2px !important;font-weight:600;font-family:'IBM Plex Sans',sans-serif;}
.stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"]{background:var(--ember) !important;border:1px solid var(--ember) !important;color:#fff !important;}
.stButton>button[kind="primary"]:hover,.stDownloadButton>button[kind="primary"]:hover{background:#963F10 !important;border-color:#963F10 !important;color:#fff !important;}
.stButton>button[kind="primary"]:disabled,.stDownloadButton>button[kind="primary"]:disabled{background:#CDBFA8 !important;border-color:#CDBFA8 !important;color:#6B6A60 !important;}
.stButton>button[kind="secondary"],.stDownloadButton>button[kind="secondary"]{background:var(--card) !important;border:1px solid var(--rule) !important;color:var(--ink) !important;}
.stButton>button[kind="secondary"]:hover,.stDownloadButton>button[kind="secondary"]:hover{border-color:var(--ember) !important;color:var(--ember) !important;}
[data-testid="stProgress"] > div{background-color:#E8E1D0 !important;border-radius:2px !important;}
[data-testid="stProgress"] div[role="progressbar"]{background-color:var(--ember) !important;background-image:none !important;}
[data-baseweb="radio"] input:checked + div{border-color:var(--ember) !important;background-color:var(--ember) !important;}
[data-baseweb="checkbox"] input:checked + div{border-color:var(--ember) !important;background-color:var(--ember) !important;border-radius:2px !important;}
[data-baseweb="checkbox"] div{border-radius:2px !important;}
a, a:visited{color:var(--ember) !important;text-decoration:underline;}
a:hover, a:active{color:#963F10 !important;}
[data-testid="stAlert"]{border-radius:2px !important;border:1px solid var(--rule) !important;background:var(--card) !important;color:var(--ink) !important;}
"""


def inject_theme() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
