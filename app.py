"""
app.py — AI vs Real Image Classifier · Enhanced Multi-Page Streamlit App
"""
import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import io
import numpy as np
import keras
from tensorflow.keras import applications
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="AI vs Real Classifier",
    page_icon="🧿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "page" not in st.session_state:
    st.session_state.page = "Detector"

IMG_SIZE         = 224
UNCERTAIN_THRESH = 0.85

MODEL_PATHS = {
    "MobileNetV2"    : "classifier_outputs/mobilenetv2.keras",
    "EfficientNetB0" : "classifier_outputs/efficientnetb0.keras",
    "NASNetMobile"   : "classifier_outputs/nasnetmobile.keras",
}
MODEL_PREPROCESS = {
    "MobileNetV2"    : applications.mobilenet_v2.preprocess_input,
    "EfficientNetB0" : applications.efficientnet.preprocess_input,
    "NASNetMobile"   : applications.nasnet.preprocess_input,
}
MODEL_INFO = {
    "MobileNetV2"    : {"params":"2.4M","speed":"Fastest","desc":"Depthwise separable convolutions. Best for real-time inference."},
    "EfficientNetB0" : {"params":"4.2M","speed":"Balanced","desc":"Compound scaling across depth, width & resolution."},
    "NASNetMobile"   : {"params":"4.4M","speed":"Robust",  "desc":"Neural Architecture Search optimised architecture."},
}

@st.cache_resource(show_spinner="Loading model…")
def load_model(name):
    path = MODEL_PATHS[name]
    if not os.path.exists(path):
        return None
    return keras.models.load_model(path)

def preprocess(pil_img, model_name):
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, 0)
    return MODEL_PREPROCESS[model_name](arr)

def predict(model, tensor):
    score = float(model.predict(tensor, verbose=0)[0][0])
    label = "Real" if score >= 0.5 else "AI Generated"
    conf  = score if score >= 0.5 else 1.0 - score
    return {
        "label"    : label,
        "conf"     : round(conf * 100, 1),
        "raw"      : round(score, 4),
        "ai_pct"   : round((1.0 - score) * 100, 1),
        "real_pct" : round(score * 100, 1),
        "uncertain": conf < UNCERTAIN_THRESH,
    }

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=JetBrains+Mono:ital,wght@0,300;0,400;0,500;1,400&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:      #07070f;
  --surf:    #0c0c1a;
  --card:    #111120;
  --card2:   #16162a;
  --border:  #1e1e33;
  --border2: #2a2a46;
  --text:    #eeeef8;
  --muted:   #50507a;
  --dim:     #7878a8;
  --accent:  #00ffd0;
  --purple:  #a06fff;
  --red:     #ff4d6d;
  --green:   #00e5a0;
  --amber:   #ffb830;
  --blue:    #4da6ff;
  --head:    'Syne', sans-serif;
  --mono:    'JetBrains Mono', monospace;
  --side:    2.4rem;
}

/* ── Streamlit chrome resets ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton, .stToolbar { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.stApp { background: var(--bg) !important; }
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"]  { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }

/* ── Streamlit column gap fix ── */
[data-testid="stHorizontalBlock"] {
  gap: 0 !important;
  align-items: stretch !important;
}

/* Hide the invisible nav-trigger buttons completely — zero height, no gap */
div[data-testid="stHorizontalBlock"]:first-of-type {
  position: absolute !important;
  width: 1px !important; height: 1px !important;
  overflow: hidden !important; opacity: 0 !important;
  pointer-events: none !important; clip: rect(0,0,0,0) !important;
}

/* ════════════════════════════════════════════
   PAGE WRAPPER
════════════════════════════════════════════ */
.pw  { padding: 0 var(--side); max-width: 1440px; margin: 0 auto; }
.pfw { padding: 0 var(--side); }

/* ════════════════════════════════════════════
   HERO
════════════════════════════════════════════ */
.hero {
  padding: 2.8rem 0 2.4rem;
  border-bottom: 1px solid var(--border);
  position: relative; overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute; top: -100px; right: -40px;
  width: 480px; height: 480px;
  background: radial-gradient(circle, rgba(0,255,208,.055) 0%, transparent 60%);
  pointer-events: none;
}
.hero-eye {
  font-family: var(--mono); font-size: 0.56rem;
  letter-spacing: 0.3em; color: var(--accent);
  text-transform: uppercase; margin-bottom: 0.65rem;
  display: flex; align-items: center; gap: 0.55rem;
}
.hero-eye::before {
  content: ''; display: inline-block; width: 22px; height: 1px; background: var(--accent);
}
.hero-h1 {
  font-family: var(--head);
  font-size: clamp(2rem, 4vw, 3.2rem);
  font-weight: 800; color: var(--text);
  letter-spacing: -0.04em; line-height: 1.04; margin-bottom: 0.7rem;
}
.hero-h1 b { color: var(--accent); }
.hero-p {
  font-family: var(--mono); font-size: 0.72rem;
  color: var(--dim); line-height: 1.9; max-width: 520px;
}

/* ════════════════════════════════════════════
   DETECTOR TWO-PANEL LAYOUT
════════════════════════════════════════════ */
.det-panel {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 10px; padding: 1.8rem;
  margin: 1.8rem 0.7rem 2.4rem;
  transition: border-color .25s, box-shadow .25s;
}
.det-panel:hover {
  border-color: rgba(0,255,208,.28);
  box-shadow: 0 0 24px rgba(0,255,208,.06), 0 0 2px rgba(0,255,208,.12);
}
.det-panel.right { background: var(--surf); }
.det-panel.right:hover {
  border-color: rgba(160,111,255,.28);
  box-shadow: 0 0 24px rgba(160,111,255,.06), 0 0 2px rgba(160,111,255,.12);
}
.dp-label {
  font-family: var(--mono); font-size: 0.5rem;
  letter-spacing: 0.24em; text-transform: uppercase;
  color: var(--accent); padding-bottom: 0.8rem;
  border-bottom: 1px solid var(--border); margin-bottom: 1.1rem;
  display: flex; gap: 0.5rem;
}
.dp-label span { color: var(--muted); }

/* model pill */
.mpill-row {
  display: flex; align-items: center; gap: 0.9rem;
  background: var(--card2); border: 1px solid var(--border2);
  border-radius: 6px; padding: 0.82rem 1rem; margin-bottom: 0.7rem;
  transition: border-color .2s, box-shadow .2s;
}
.mpill-row:hover {
  border-color: rgba(0,255,208,.3);
  box-shadow: 0 0 12px rgba(0,255,208,.07);
}
.mpill-dot {
  width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0;
  background: var(--accent); box-shadow: 0 0 8px var(--accent);
  animation: glow 2.4s ease-in-out infinite;
}
.mpill-name {
  font-family: var(--head); font-size: 0.9rem;
  font-weight: 700; color: var(--text); flex: 1;
}
.mpill-tag {
  font-family: var(--mono); font-size: 0.48rem;
  letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--accent); border: 1px solid rgba(0,255,208,.25);
  border-radius: 3px; padding: 0.16rem 0.48rem;
  background: rgba(0,255,208,.05);
}
.mdesc {
  font-family: var(--mono); font-size: 0.67rem;
  color: var(--muted); line-height: 1.75; margin-bottom: 1.4rem;
}

/* uploaded image */
.img-box {
  border: 1px solid var(--border2); border-radius: 6px; overflow: hidden;
  background: var(--bg); margin: 0.6rem 0;
  transition: border-color .2s, box-shadow .2s;
}
.img-box:hover {
  border-color: rgba(0,255,208,.3);
  box-shadow: 0 0 16px rgba(0,255,208,.08);
}
.img-meta {
  font-family: var(--mono); font-size: 0.54rem;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--muted); padding: 0.45rem 0.8rem;
  border-top: 1px solid var(--border); display: flex; gap: 1rem;
}

/* result card */
.result {
  border: 1px solid var(--border2); border-radius: 8px;
  background: var(--card2); padding: 1.6rem;
  position: relative; overflow: hidden;
  animation: up .3s ease; margin-bottom: 0.9rem;
  transition: border-color .25s, box-shadow .25s;
}
.result:hover {
  box-shadow: 0 0 20px rgba(0,0,0,.3);
}
@keyframes up { from { opacity:0; transform:translateY(8px);} to {opacity:1;transform:translateY(0);} }
.result::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; }
.result.real::before  { background: linear-gradient(90deg,transparent,var(--green),transparent); }
.result.fake::before  { background: linear-gradient(90deg,transparent,var(--red),transparent); }
.result.unsure::before{ background: linear-gradient(90deg,transparent,var(--amber),transparent); }
.result.real:hover  { border-color: rgba(0,229,160,.3); box-shadow: 0 0 20px rgba(0,229,160,.07); }
.result.fake:hover  { border-color: rgba(255,77,109,.3); box-shadow: 0 0 20px rgba(255,77,109,.07); }
.result.unsure:hover{ border-color: rgba(255,184,48,.3); box-shadow: 0 0 20px rgba(255,184,48,.07); }

.verd-lbl { font-family:var(--mono); font-size:.48rem; letter-spacing:.24em; color:var(--muted); text-transform:uppercase; }
.verd { font-family:var(--head); font-size:2.5rem; font-weight:800; letter-spacing:-.04em; line-height:1; margin:.25rem 0; }
.verd.real{color:var(--green);} .verd.fake{color:var(--red);} .verd.unsure{color:var(--amber);}
.verd-sub { font-family:var(--mono); font-size:.66rem; color:var(--muted); margin-bottom:1.3rem; }

.brow { display:flex; justify-content:space-between; font-family:var(--mono); font-size:.58rem; color:var(--muted); margin-bottom:.3rem; }
.btrack { height:4px; background:var(--border); border-radius:4px; overflow:hidden; margin-bottom:.65rem; }
.bfill  { height:100%; border-radius:4px; transition:width .6s cubic-bezier(.4,0,.2,1); }
.bg { background:linear-gradient(90deg,var(--green),#00ffb3); }
.br { background:linear-gradient(90deg,var(--red),#ff8099); }

.scores { display:grid; grid-template-columns:1fr 1fr 1fr; gap:.55rem; margin-top:.9rem; }
.sc {
  background:var(--surf); border:1px solid var(--border); border-radius:5px; padding:.65rem; text-align:center;
  transition: border-color .2s, box-shadow .2s;
}
.sc:hover { border-color: rgba(0,255,208,.25); box-shadow: 0 0 10px rgba(0,255,208,.06); }
.sc-v { font-family:var(--mono); font-size:.9rem; font-weight:500; color:var(--text); display:block; }
.sc-l { font-family:var(--mono); font-size:.47rem; letter-spacing:.14em; color:var(--muted); text-transform:uppercase; display:block; margin-top:.22rem; }

.warn-badge {
  display:inline-flex; align-items:center; gap:.4rem;
  font-family:var(--mono); font-size:.58rem; color:var(--amber);
  border:1px solid rgba(255,184,48,.3); background:rgba(255,184,48,.06);
  border-radius:4px; padding:.28rem .65rem; margin-bottom:.9rem;
}
.info-card {
  background:var(--card); border:1px solid var(--border);
  border-radius:6px; padding:1rem 1.1rem;
  transition: border-color .2s, box-shadow .2s;
}
.info-card:hover { border-color: rgba(0,255,208,.2); box-shadow: 0 0 12px rgba(0,255,208,.05); }
.info-card .card-lbl { font-family:var(--mono); font-size:.48rem; letter-spacing:.22em; color:var(--muted); text-transform:uppercase; margin-bottom:.45rem; }
.info-card .card-p   { font-family:var(--mono); font-size:.68rem; color:var(--dim); line-height:1.85; }

.empty {
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  min-height:360px; text-align:center; gap:.8rem;
}
.empty-icon { font-size:3rem; opacity:.1; }
.empty-t { font-family:var(--mono); font-size:.66rem; letter-spacing:.2em; color:var(--muted); text-transform:uppercase; }
.empty-s { font-family:var(--mono); font-size:.6rem; color:var(--muted); max-width:210px; line-height:1.9; }

/* ════════════════════════════════════════════
   SHARED COMPONENTS
════════════════════════════════════════════ */
.sec {
  font-family:var(--head); font-size:1.1rem; font-weight:700;
  color:var(--text); letter-spacing:-.02em; margin:2rem 0 1.1rem;
  display:flex; align-items:center; gap:.8rem;
}
.sec::after { content:''; flex:1; height:1px; background:var(--border); }
.hr { height:1px; background:var(--border); margin:2.2rem 0; }

.card {
  background:var(--card); border:1px solid var(--border);
  border-radius:8px; padding:1.4rem 1.5rem; margin-bottom:.9rem;
  transition: border-color .22s, box-shadow .22s, transform .22s;
}
.card:hover {
  border-color: rgba(0,255,208,.22);
  box-shadow: 0 0 22px rgba(0,255,208,.06), 0 4px 16px rgba(0,0,0,.25);
  transform: translateY(-2px);
}
.card.a{ border-top:2px solid var(--accent); }
.card.a:hover { border-color: rgba(0,255,208,.4); box-shadow: 0 0 22px rgba(0,255,208,.1), 0 4px 16px rgba(0,0,0,.25); }
.card.p{ border-top:2px solid var(--purple); }
.card.p:hover { border-color: rgba(160,111,255,.4); box-shadow: 0 0 22px rgba(160,111,255,.1), 0 4px 16px rgba(0,0,0,.25); }
.card.g{ border-top:2px solid var(--green); }
.card.g:hover { border-color: rgba(0,229,160,.4); box-shadow: 0 0 22px rgba(0,229,160,.1), 0 4px 16px rgba(0,0,0,.25); }
.card.r{ border-top:2px solid var(--red); }
.card.r:hover { border-color: rgba(255,77,109,.4); box-shadow: 0 0 22px rgba(255,77,109,.1), 0 4px 16px rgba(0,0,0,.25); }
.card.am{ border-top:2px solid var(--amber); }
.card.am:hover { border-color: rgba(255,184,48,.4); box-shadow: 0 0 22px rgba(255,184,48,.1), 0 4px 16px rgba(0,0,0,.25); }
.card.b{ border-top:2px solid var(--blue); }
.card.b:hover { border-color: rgba(77,166,255,.4); box-shadow: 0 0 22px rgba(77,166,255,.1), 0 4px 16px rgba(0,0,0,.25); }

.card-lbl { font-family:var(--mono); font-size:.5rem; letter-spacing:.24em; color:var(--muted); text-transform:uppercase; margin-bottom:.55rem; }
.card-h   { font-family:var(--head); font-size:.96rem; font-weight:700; color:var(--text); margin-bottom:.45rem; }
.card-p   { font-family:var(--mono); font-size:.7rem; color:var(--dim); line-height:1.9; }

.stats { display:grid; grid-template-columns:repeat(4,1fr); gap:.9rem; margin:1.8rem 0; }
.stat {
  background:var(--card); border:1px solid var(--border); border-radius:8px;
  padding:1.3rem; text-align:center;
  transition: border-color .22s, box-shadow .22s, transform .22s;
  position:relative; overflow:hidden;
}
.stat:hover {
  border-color: rgba(0,255,208,.3);
  box-shadow: 0 0 28px rgba(0,255,208,.09), 0 4px 18px rgba(0,0,0,.3);
  transform: translateY(-3px);
}
.stat-v { font-family:var(--head); font-size:2.1rem; font-weight:800; line-height:1; display:block; margin-bottom:.35rem; }
.stat-l { font-family:var(--mono); font-size:.5rem; letter-spacing:.18em; color:var(--muted); text-transform:uppercase; }

.tags { display:flex; gap:.4rem; flex-wrap:wrap; margin-top:.8rem; }
.tag  { font-family:var(--mono); font-size:.52rem; letter-spacing:.1em; text-transform:uppercase; border-radius:3px; padding:.18rem .55rem; border:1px solid; }
.ta  { color:var(--accent); border-color:rgba(0,255,208,.3);  background:rgba(0,255,208,.06); }
.tp  { color:var(--purple); border-color:rgba(160,111,255,.3); background:rgba(160,111,255,.06); }
.tg  { color:var(--green);  border-color:rgba(0,229,160,.3);   background:rgba(0,229,160,.06); }
.tam { color:var(--amber);  border-color:rgba(255,184,48,.3);  background:rgba(255,184,48,.06); }
.tb  { color:var(--blue);   border-color:rgba(77,166,255,.3);  background:rgba(77,166,255,.06); }

.step {
  display:flex; gap:1.3rem; padding:1.3rem 1.4rem; background:var(--card);
  border:1px solid var(--border); border-radius:8px;
  margin-bottom:.85rem; align-items:flex-start;
  transition: border-color .22s, box-shadow .22s, transform .22s;
}
.step:hover {
  border-color: rgba(0,255,208,.28);
  box-shadow: 0 0 20px rgba(0,255,208,.07), 0 4px 14px rgba(0,0,0,.25);
  transform: translateX(4px);
}
.step-n { font-family:var(--head); font-size:1.9rem; font-weight:800; color:var(--border2); line-height:1; min-width:2.2rem; text-align:center; padding-top:.05rem; }
.step-h { font-family:var(--head); font-size:.92rem; font-weight:700; color:var(--text); margin-bottom:.35rem; }
.step-b { font-family:var(--mono); font-size:.69rem; color:var(--dim); line-height:1.9; }

.code {
  background:#050510; border:1px solid var(--border);
  border-left:3px solid var(--accent); border-radius:6px;
  padding:1.2rem 1.4rem; font-family:var(--mono); font-size:.67rem;
  color:#8080b0; line-height:2.1; overflow-x:auto; white-space:pre; margin:1rem 0;
  transition: border-color .2s, box-shadow .2s;
}
.code:hover {
  border-color: rgba(0,255,208,.3);
  box-shadow: 0 0 18px rgba(0,255,208,.05);
}
.kw{color:var(--purple);} .fn{color:var(--accent);} .st{color:var(--green);} .cm{color:#35355a;} .nm{color:var(--amber);}

.tbl { width:100%; border-collapse:collapse; }
.tbl td { font-family:var(--mono); font-size:.68rem; padding:.62rem .7rem; border-bottom:1px solid var(--border); vertical-align:top; }
.tbl td:first-child { color:var(--muted); width:36%; font-size:.57rem; letter-spacing:.07em; text-transform:uppercase; }
.tbl td:last-child  { color:var(--text); }
.tbl tr:hover td { background: rgba(0,255,208,.03); }

/* footer */
.foot {
  border-top:1px solid var(--border); padding:1.1rem var(--side);
  display:flex; justify-content:space-between; align-items:center;
  background:var(--surf); margin-top:3rem;
}
.foot-l { font-family:var(--mono); font-size:.53rem; letter-spacing:.14em; color:var(--muted); text-transform:uppercase; }
.foot-r a { font-family:var(--mono); font-size:.55rem; color:var(--accent); text-decoration:none; }

/* ── Streamlit widget skins ── */
.stSelectbox > div > div {
  background:var(--card2) !important; border:1px solid var(--border2) !important;
  border-radius:5px !important; color:var(--text) !important;
  font-family:var(--mono) !important; font-size:.82rem !important;
  transition: border-color .2s, box-shadow .2s !important;
}
.stSelectbox > div > div:hover {
  border-color: rgba(0,255,208,.35) !important;
  box-shadow: 0 0 12px rgba(0,255,208,.08) !important;
}
.stSelectbox label { color:var(--muted) !important; font-family:var(--mono) !important; font-size:.56rem !important; letter-spacing:.16em !important; text-transform:uppercase !important; }
.stFileUploader > div {
  border:1px dashed var(--border2) !important; border-radius:7px !important;
  background:var(--card2) !important;
  transition: border-color .2s, box-shadow .2s !important;
}
.stFileUploader > div:hover {
  border-color:var(--accent) !important;
  box-shadow: 0 0 18px rgba(0,255,208,.1) !important;
}
.stFileUploader label { color:var(--muted) !important; font-family:var(--mono) !important; font-size:.56rem !important; letter-spacing:.16em !important; text-transform:uppercase !important; }
.stImage img { border-radius:0 !important; display:block !important; }
.stSpinner > div { border-top-color:var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# NAVBAR — pure HTML component with postMessage nav
# ════════════════════════════════════════════════════════════
NAV_PAGES = ["Detector", "About", "Tech Stack", "How It Works"]
page = st.session_state.page

# Render the full navbar as a self-contained HTML component.
# Clicking a nav link sends a postMessage to the parent Streamlit window,
# which is caught by a small <script> injected via st.markdown.
def build_nav_links(current):
    html = ""
    for p in NAV_PAGES:
        active = "active" if p == current else ""
        html += f'<a class="tnav-link {active}" onclick="navigate(\'{p}\')">{p.upper()}</a>\n'
    return html

components.html(f"""
<!DOCTYPE html>
<html>
<head>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background: rgba(7,7,15,0.95); overflow: hidden; }}
  nav {{
    height: 58px; display: flex; align-items: center;
    padding: 0 2.4rem; gap: 2rem;
    background: rgba(7,7,15,0.92);
    backdrop-filter: blur(18px) saturate(160%);
    border-bottom: 1px solid #1e1e33;
    position: relative;
  }}
  nav::after {{
    content: ''; position: absolute; bottom: -1px; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(0,255,208,.18) 50%, transparent 100%);
  }}
  .logo {{
    display: flex; align-items: center; gap: 0.5rem;
    font-family: 'Syne', sans-serif; font-size: 1.12rem; font-weight: 800;
    color: #eeeef8; letter-spacing: -0.03em; white-space: nowrap; flex-shrink: 0;
  }}
  .logo em {{ color: #00ffd0; font-style: normal; }}
  .dot {{
    width: 7px; height: 7px; border-radius: 50%;
    background: #00ffd0; box-shadow: 0 0 10px #00ffd0;
    animation: glow 2.4s ease-in-out infinite; flex-shrink: 0;
  }}
  @keyframes glow {{
    0%,100% {{ box-shadow: 0 0 5px #00ffd0; }}
    50%      {{ box-shadow: 0 0 16px #00ffd0, 0 0 28px rgba(0,255,208,.25); }}
  }}
  .div {{ width:1px; height:20px; background:#2a2a46; flex-shrink:0; }}
  .links {{ display:flex; align-items:center; gap:0.2rem; flex:1; }}
  .tnav-link {{
    font-family: 'JetBrains Mono', monospace; font-size: 0.67rem;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: #50507a; text-decoration: none;
    border: 1px solid transparent; border-radius: 4px;
    padding: 0.38rem 0.95rem;
    transition: color .16s, background .16s, border-color .16s;
    cursor: pointer; white-space: nowrap; user-select: none;
  }}
  .tnav-link:hover {{ color: #eeeef8; background: #111120; border-color: #2a2a46; }}
  .tnav-link.active {{
    color: #00ffd0; background: rgba(0,255,208,.07);
    border-color: rgba(0,255,208,.22);
  }}
  .right {{ margin-left: auto; display:flex; align-items:center; gap:0.7rem; flex-shrink:0; }}
  .badge {{
    font-family: 'JetBrains Mono', monospace; font-size: 0.5rem; letter-spacing: 0.16em;
    color: #50507a; border: 1px solid #2a2a46; border-radius: 3px;
    padding: 0.2rem 0.5rem; text-transform: uppercase;
  }}
  .gh {{
    font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; letter-spacing: 0.06em;
    color: #00ffd0; border: 1px solid rgba(0,255,208,.24); border-radius: 4px;
    padding: 0.32rem 0.75rem; text-decoration: none;
    background: rgba(0,255,208,.04); transition: background .16s, border-color .16s;
  }}
  .gh:hover {{ background: rgba(0,255,208,.12); border-color: rgba(0,255,208,.5); }}
</style>
</head>
<body>
<nav>
  <span class="logo"><span class="dot"></span>AI<em>vs</em>Real</span>
  <div class="div"></div>
  <div class="links">
    {build_nav_links(page)}
  </div>
  <div class="right">
    <span class="badge">TF 2.21</span>
    <a class="gh" href="https://github.com/nikhilsai0803/ai-vs-real-image-classifier" target="_blank">GitHub ↗</a>
  </div>
</nav>
<script>
  function navigate(p) {{
    window.parent.postMessage({{type: "streamlit:setComponentValue", value: p}}, "*");
  }}
</script>
</body>
</html>
""", height=60, scrolling=False)

# Listen for postMessage from the nav component
# We use a hidden st.text_input driven by JS injection as the bridge
st.markdown("""
<script>
window.addEventListener('message', function(e) {
  if (e.data && e.data.type === 'streamlit:setComponentValue') {
    // Find all buttons and click the matching one
    var target = e.data.value;
    var buttons = window.parent.document.querySelectorAll('button[kind="secondary"], button');
    buttons.forEach(function(btn) {
      if (btn.innerText.trim().toLowerCase() === target.toLowerCase()) {
        btn.click();
      }
    });
  }
});
</script>
""", unsafe_allow_html=True)

# Hidden session-state nav buttons (invisible, triggered by postMessage above)
# Rendered with zero height so they don't take space
st.markdown("""
<style>
div[data-testid="stHorizontalBlock"]:has(button[data-nav="true"]) {
  position: absolute !important;
  opacity: 0 !important;
  pointer-events: none !important;
  height: 0 !important;
  overflow: hidden !important;
}
</style>
""", unsafe_allow_html=True)

# These buttons are the actual Streamlit triggers — hidden via CSS
_nav_cols = st.columns(len(NAV_PAGES))
for _col, _nav_page in zip(_nav_cols, NAV_PAGES):
    if _col.button(_nav_page, key=f"nav_{_nav_page}"):
        st.session_state.page = _nav_page
        st.rerun()

page = st.session_state.page

# ════════════════════════════════════════════════════════════
# HERO
# ════════════════════════════════════════════════════════════
HEROES = {
    "Detector"   : ("Live Inference Engine",  "Image <b>Detector</b>",        "Upload any image. The selected model classifies it as AI-generated or a real photograph with a full confidence breakdown."),
    "About"      : ("Project Overview",       "About <b>This Project</b>",    "A deep learning binary classifier distinguishing AI-generated artwork from real photographs using transfer learning on three pretrained architectures."),
    "Tech Stack" : ("Tools & Libraries",      "Tech <b>Stack</b>",            "Every library, framework, and tool used to build, train, evaluate, and deploy this project."),
    "How It Works":("Technical Deep Dive",    "How It <b>Works</b>",          "The full pipeline — from raw images on disk to a confident prediction — explained step by step."),
}
eye, h1, sub = HEROES.get(page, HEROES["Detector"])
st.markdown(f"""
<div class="pfw">
  <div class="hero">
    <div class="hero-eye">{eye}</div>
    <div class="hero-h1">{h1}</div>
    <div class="hero-p">{sub}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE 1 — DETECTOR
# ════════════════════════════════════════════════════════════
if page == "Detector":
    col_l, col_r = st.columns(2, gap="medium")

    with col_l:
        st.markdown('<div class="det-panel">', unsafe_allow_html=True)
        st.markdown('<div class="dp-label">01 · Select Model <span>— architecture</span></div>', unsafe_allow_html=True)

        selected = st.selectbox("model", list(MODEL_PATHS.keys()), label_visibility="collapsed")
        info = MODEL_INFO[selected]
        st.markdown(f"""
        <div class="mpill-row">
          <div class="mpill-dot"></div>
          <div class="mpill-name">{selected}</div>
          <span class="mpill-tag">Fine-tuned</span>
          <span class="mpill-tag">ImageNet</span>
        </div>
        <div class="mdesc">{info['desc']}&nbsp;&nbsp;<span style="color:var(--text)">{info['params']} params · {info['speed']}</span></div>
        """, unsafe_allow_html=True)

        model = load_model(selected)
        if model is None:
            st.error(f"**Model not found:** `{MODEL_PATHS[selected]}`\n\nRun the training notebook to generate `classifier_outputs/`.")
            st.stop()

        st.markdown('<div class="dp-label" style="margin-top:.6rem;">02 · Upload Image <span>— jpg · png · webp</span></div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("img", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")

        if uploaded:
            pil_img = Image.open(io.BytesIO(uploaded.read()))
            w, h = pil_img.size
            st.markdown('<div class="img-box">', unsafe_allow_html=True)
            st.image(pil_img, use_container_width=True)
            st.markdown(f'<div class="img-meta"><span>{uploaded.name}</span><span>{w}×{h}px</span><span>{pil_img.mode}</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="det-panel right">', unsafe_allow_html=True)
        st.markdown('<div class="dp-label">03 · Result <span>— classification output</span></div>', unsafe_allow_html=True)

        if uploaded:
            with st.spinner("Analysing…"):
                tensor = preprocess(pil_img, selected)
                res    = predict(model, tensor)

            label, conf    = res["label"], res["conf"]
            raw, uncertain = res["raw"], res["uncertain"]
            ai_pct, rpct   = res["ai_pct"], res["real_pct"]

            if uncertain:         cls, icon, vc = "unsure", "⚠️", "unsure"
            elif "Real" in label: cls, icon, vc = "real",   "✅", "real"
            else:                 cls, icon, vc = "fake",   "🤖", "fake"

            warn_html = '<div class="warn-badge">⚠ LOW CONFIDENCE — result may be unreliable</div>' if uncertain else ""

            st.markdown(f"""
            <div class="result {cls}">
              <div class="verd-lbl">Verdict</div>
              <div style="font-size:2rem;margin:.25rem 0;">{icon}</div>
              <div class="verd {vc}">{label}</div>
              <div class="verd-sub">Confidence: {conf}% &nbsp;·&nbsp; Raw sigmoid: {raw:.4f}</div>
              {warn_html}
              <div class="brow"><span>🤖 AI-Generated</span><span>{ai_pct}%</span></div>
              <div class="btrack"><div class="bfill br" style="width:{ai_pct}%"></div></div>
              <div class="brow"><span>📷 Real Photo</span><span>{rpct}%</span></div>
              <div class="btrack"><div class="bfill bg" style="width:{rpct}%"></div></div>
              <div class="scores">
                <div class="sc"><span class="sc-v">{raw:.4f}</span><span class="sc-l">Raw Score</span></div>
                <div class="sc"><span class="sc-v">{conf}%</span><span class="sc-l">Confidence</span></div>
                <div class="sc"><span class="sc-v" style="font-size:.6rem;">{selected}</span><span class="sc-l">Model</span></div>
              </div>
            </div>
            <div class="info-card">
              <div class="card-lbl">How to read this</div>
              <div class="card-p">Score &gt; 0.5 → <span style="color:var(--green)">Real Photo</span>&nbsp;|&nbsp;Score &lt; 0.5 → <span style="color:var(--red)">AI-Generated</span><br>
              Confidence below <strong style="color:var(--amber)">85%</strong> is flagged as uncertain.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty">
              <div class="empty-icon">🧿</div>
              <div class="empty-t">Awaiting Image</div>
              <div class="empty-s">Upload a JPG, PNG, or WEBP in the left panel to run inference</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE 2 — ABOUT
# ════════════════════════════════════════════════════════════
elif page == "About":
    st.markdown('<div class="pw">', unsafe_allow_html=True)

    st.markdown("""
    <div class="stats">
      <div class="stat"><span class="stat-v" style="color:var(--accent)">~4.7K</span><span class="stat-l">Training Images</span></div>
      <div class="stat"><span class="stat-v" style="color:var(--purple)">3</span><span class="stat-l">Models Trained</span></div>
      <div class="stat"><span class="stat-v" style="color:var(--green)">2</span><span class="stat-l">Training Phases</span></div>
      <div class="stat"><span class="stat-v" style="color:var(--amber)">85%</span><span class="stat-l">Confidence Threshold</span></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("""
        <div class="card a">
          <div class="card-lbl">The Problem</div>
          <div class="card-h">Why Does This Matter?</div>
          <div class="card-p">AI-generated images have become indistinguishable from real photographs. Tools like Midjourney, DALL·E, and Stable Diffusion raise concerns around misinformation, copyright, and digital trust.<br><br>This classifier detects differences at a feature level — patterns the human eye cannot perceive.</div>
          <div class="tags"><span class="tag ta">Binary Classification</span><span class="tag tp">Computer Vision</span></div>
        </div>
        <div class="card g">
          <div class="card-lbl">Dataset</div>
          <div class="card-h">Training Data</div>
          <div class="card-p">
            <table class="tbl">
              <tr><td>Source</td><td>Kaggle · tristanzhang32</td></tr>
              <tr><td>AI Images</td><td>~2,300 AI-generated artworks</td></tr>
              <tr><td>Real Images</td><td>~2,400 real photographs</td></tr>
              <tr><td>Split</td><td>70% Train · 15% Val · 15% Test</td></tr>
              <tr><td>Validation</td><td>Corrupt files removed before training</td></tr>
            </table>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card p">
          <div class="card-lbl">Approach</div>
          <div class="card-h">Transfer Learning Strategy</div>
          <div class="card-p">Models pre-trained on ImageNet already know how to detect edges, textures, shapes, and complex visual patterns.<br><br>We fine-tune them to learn AI-art artefacts: synthetic gradients, GAN noise signatures, unnatural smoothness — things invisible to human eyes.</div>
          <div class="tags"><span class="tag tp">Transfer Learning</span><span class="tag tg">Fine-tuning</span><span class="tag tam">ImageNet</span></div>
        </div>
        <div class="card am">
          <div class="card-lbl">Key Fixes Applied</div>
          <div class="card-h">What Made It Work</div>
          <div class="card-p">
            <table class="tbl">
              <tr><td>Preprocessing</td><td>Each model uses its own <code style="color:var(--accent)">preprocess_input</code> — no manual /255</td></tr>
              <tr><td>Fine-tuning</td><td>Only last 30 layers unfrozen — not the full base</td></tr>
              <tr><td>Callbacks</td><td>Fresh EarlyStopping per training phase</td></tr>
              <tr><td>Validation</td><td>Corrupt images removed at startup</td></tr>
            </table>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="hr"></div><div class="sec">Project Goals</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown("""<div class="card"><div style="font-size:1.6rem;margin-bottom:.5rem;">🎯</div>
        <div class="card-h">Accurate Classification</div>
        <div class="card-p">Learn genuine visual features — not brightness or colour shortcuts — to reliably separate AI from real.</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card"><div style="font-size:1.6rem;margin-bottom:.5rem;">⚡</div>
        <div class="card-h">Lightweight & Fast</div>
        <div class="card-p">Mobile-scale architectures so inference runs in seconds even on CPU — no GPU required.</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="card"><div style="font-size:1.6rem;margin-bottom:.5rem;">🌐</div>
        <div class="card-h">Deployable UI</div>
        <div class="card-p">A polished web interface anyone can use — no code required. Upload, click, get your answer.</div></div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE 3 — TECH STACK
# ════════════════════════════════════════════════════════════
elif page == "Tech Stack":
    st.markdown('<div class="pw">', unsafe_allow_html=True)
    st.markdown('<div class="sec">Core ML Framework</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown("""<div class="card a"><div style="font-size:1.4rem;margin-bottom:.45rem;">🧠</div>
        <div class="card-h">TensorFlow 2.21</div>
        <div class="card-p">Full ML lifecycle — <code style="color:var(--accent)">tf.data</code> pipelines, training loops, model saving and serving.</div>
        <div class="tags"><span class="tag ta">Core</span><span class="tag tp">Google</span></div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card p"><div style="font-size:1.4rem;margin-bottom:.45rem;">🔷</div>
        <div class="card-h">Keras</div>
        <div class="card-p">High-level API. Defines model architecture, compiles, runs training. The <code style="color:var(--accent)">applications</code> module provides all three pretrained bases.</div>
        <div class="tags"><span class="tag tp">API</span><span class="tag ta">Built-in</span></div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="card g"><div style="font-size:1.4rem;margin-bottom:.45rem;">📦</div>
        <div class="card-h">NumPy</div>
        <div class="card-p">Converts PIL images to float32 arrays, stacks batches, post-processes raw sigmoid scores for display.</div>
        <div class="tags"><span class="tag tg">Numerical</span></div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="hr"></div><div class="sec">Pretrained Architectures</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown("""<div class="card b"><div style="font-size:1.3rem;margin-bottom:.45rem;">📱</div>
        <div class="card-h">MobileNetV2</div>
        <div class="card-p"><table class="tbl">
          <tr><td>Params</td><td>2.4M total</td></tr>
          <tr><td>Input</td><td>[0,255] → [-1, 1]</td></tr>
          <tr><td>Design</td><td>Depthwise separable + inverted residuals</td></tr>
          <tr><td>Best for</td><td>Real-time mobile inference</td></tr>
        </table></div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card a"><div style="font-size:1.3rem;margin-bottom:.45rem;">⚖️</div>
        <div class="card-h">EfficientNetB0</div>
        <div class="card-p"><table class="tbl">
          <tr><td>Params</td><td>4.2M total</td></tr>
          <tr><td>Input</td><td>[0,255] → normalised internally</td></tr>
          <tr><td>Design</td><td>Compound scaling: depth, width, resolution</td></tr>
          <tr><td>Best for</td><td>Highest accuracy per parameter</td></tr>
        </table></div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="card p"><div style="font-size:1.3rem;margin-bottom:.45rem;">🔬</div>
        <div class="card-h">NASNetMobile</div>
        <div class="card-p"><table class="tbl">
          <tr><td>Params</td><td>4.4M total</td></tr>
          <tr><td>Input</td><td>[0,255] → [-1, 1]</td></tr>
          <tr><td>Design</td><td>Neural Architecture Search by Google</td></tr>
          <tr><td>Best for</td><td>Robustness &amp; generalisation</td></tr>
        </table></div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="hr"></div><div class="sec">Supporting Libraries & Deployment</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("""<div class="card"><div class="card-lbl">Data & Visualisation</div>
        <table class="tbl">
          <tr><td>scikit-learn</td><td>train_test_split, classification_report, confusion_matrix</td></tr>
          <tr><td>Matplotlib</td><td>Training curves, distributions, sample grids</td></tr>
          <tr><td>Seaborn</td><td>Styled confusion matrix heatmaps</td></tr>
          <tr><td>Pillow</td><td>Image loading and format conversion in the app</td></tr>
        </table></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card"><div class="card-lbl">App & Deployment</div>
        <table class="tbl">
          <tr><td>Streamlit</td><td>Web app framework — all UI and state management</td></tr>
          <tr><td>Streamlit Cloud</td><td>Free hosting — one-click deploy from GitHub</td></tr>
          <tr><td>packages.txt</td><td>System deps (libgl1) for Streamlit Cloud</td></tr>
          <tr><td>requirements.txt</td><td>All Python dependencies pinned</td></tr>
        </table></div>""", unsafe_allow_html=True)

    st.markdown('<div class="hr"></div><div class="sec">requirements.txt</div>', unsafe_allow_html=True)
    st.markdown("""<div class="code"><span class="cm"># pip install -r requirements.txt</span>

<span class="kw">tensorflow</span>==<span class="nm">2.21.0</span>
<span class="kw">streamlit</span>&gt;=<span class="nm">1.35.0</span>
<span class="kw">numpy</span>&gt;=<span class="nm">1.24.0</span>
<span class="kw">Pillow</span>&gt;=<span class="nm">10.0.0</span>
<span class="kw">scikit-learn</span>&gt;=<span class="nm">1.3.0</span>
<span class="kw">matplotlib</span>&gt;=<span class="nm">3.7.0</span>
<span class="kw">seaborn</span>&gt;=<span class="nm">0.12.0</span></div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE 4 — HOW IT WORKS
# ════════════════════════════════════════════════════════════
elif page == "How It Works":
    st.markdown('<div class="pw">', unsafe_allow_html=True)
    st.markdown('<div class="sec">Training Pipeline</div>', unsafe_allow_html=True)

    steps = [
        ("Image Collection & Validation",
         "All images are scanned recursively from <code style='color:var(--accent)'>AiArtData/</code> and <code style='color:var(--accent)'>RealArt/</code>. Before training begins, every file is decoded by TensorFlow. Corrupt files, truncated JPEGs, and images smaller than 10×10px are <strong style='color:var(--red)'>detected and removed</strong> so they never cause silent failures mid-training."),
        ("Data Split & tf.data Pipeline",
         "Images are shuffled and split 70/15/15 (train/val/test). A <code style='color:var(--accent)'>tf.data.Dataset</code> pipeline loads images lazily on demand. Training gets light augmentation: random flip, ±10% brightness, ±15% contrast. Critical fix: images stay as <code style='color:var(--accent)'>float32 in [0, 255]</code> — <strong style='color:var(--red)'>no /255 division</strong>."),
        ("Phase 1 — Head Training",
         "The base model is <strong>fully frozen</strong>. Only the classification head trains: GlobalAveragePooling → BatchNorm → Dense(256) → Dropout(0.4) → Dense(64) → Dropout(0.2) → Sigmoid. Learning rate: 1e-3. EarlyStopping with patience=3. Typically converges in 5–9 epochs, reaching ~70–75% validation accuracy."),
        ("Phase 2 — Surgical Fine-tuning",
         "Only the <strong>last 30 layers</strong> of the base model are unfrozen. Learning rate drops to 1e-5. A <strong>fresh EarlyStopping callback</strong> is created (the Phase 1 callback had stale state). This phase teaches AI-art-specific patterns: GAN noise, synthetic gradients, unnatural smoothness."),
        ("Evaluation & Saving",
         "All models are evaluated on the held-out test set. Confusion matrices, precision/recall/F1, and accuracy are reported. Every graph and model is saved to <code style='color:var(--accent)'>classifier_outputs/</code> as <code style='color:var(--accent)'>.keras</code> files for serving in this app."),
    ]
    for i, (title, body) in enumerate(steps, 1):
        st.markdown(f"""
        <div class="step">
          <div class="step-n">0{i}</div>
          <div><div class="step-h">{title}</div><div class="step-b">{body}</div></div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="hr"></div><div class="sec">Inference Pipeline</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("""<div class="card a">
        <div class="card-lbl">What Happens When You Upload</div>
        <div class="card-p" style="line-height:2.3;">
          1. PIL opens image from memory — no disk write<br>
          2. Resized to 224×224 px<br>
          3. Cast to float32 in [0, 255]<br>
          4. Model-specific <code style="color:var(--accent)">preprocess_input</code> scales it<br>
          5. Model outputs a sigmoid score in [0, 1]<br>
          6. Score ≥ 0.5 → <span style="color:var(--green)">Real</span>&nbsp;|&nbsp;Score &lt; 0.5 → <span style="color:var(--red)">AI</span><br>
          7. Confidence = how far score is from 0.5<br>
          8. Result shown with bar breakdown
        </div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card r">
        <div class="card-lbl">The Critical Bug We Fixed</div>
        <div class="card-p">
          The original code divided by 255 <em>before</em> calling <code style="color:var(--accent)">preprocess_input</code>.<br><br>
          Each model expects [0, 255] as input:<br>
          <span style="color:var(--green)">MobileNetV2</span> → scales to [-1, 1]<br>
          <span style="color:var(--accent)">EfficientNetB0</span> → normalises internally<br>
          <span style="color:var(--purple)">NASNetMobile</span> → scales to [-1, 1]<br><br>
          Dividing first sent wrong values through the scaler — producing garbage activations. All models predicted only "Real" at <strong style="color:var(--red)">49% accuracy</strong>.
        </div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="hr"></div><div class="sec">Key Code Fixes</div>', unsafe_allow_html=True)
    st.markdown("""<div class="code"><span class="cm"># ❌ WRONG — caused 49% accuracy</span>
<span class="kw">def</span> <span class="fn">parse_image</span>(path, label):
    img = tf.<span class="fn">cast</span>(img, tf.float32) / <span class="nm">255.0</span>  <span class="cm"># ← breaks preprocess_input</span>
    <span class="kw">return</span> img, label

<span class="cm"># ✅ CORRECT — cast only, let each model handle its own scaling</span>
<span class="kw">def</span> <span class="fn">parse_image</span>(path, label):
    img = tf.<span class="fn">cast</span>(img, tf.float32)  <span class="cm"># NO /255</span>
    <span class="kw">return</span> img, label

<span class="cm"># ❌ WRONG — unfreezes ALL layers, destroys ImageNet weights</span>
model.layers[<span class="nm">1</span>].trainable = <span class="kw">True</span>

<span class="cm"># ✅ CORRECT — only last 30 layers + fresh EarlyStopping for Phase 2</span>
<span class="kw">for</span> layer <span class="kw">in</span> base.layers:        layer.trainable = <span class="kw">False</span>
<span class="kw">for</span> layer <span class="kw">in</span> base.layers[-<span class="nm">30</span>:]:  layer.trainable = <span class="kw">True</span>
cb_p2 = [keras.callbacks.<span class="fn">EarlyStopping</span>(monitor=<span class="st">'val_loss'</span>, patience=<span class="nm">4</span>)]</div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div class="foot">
  <div class="foot-l">AI vs Real Image Classifier &nbsp;·&nbsp; TensorFlow &nbsp;·&nbsp; Transfer Learning &nbsp;·&nbsp; Streamlit</div>
  <div class="foot-r"><a href="https://github.com/nikhilsai0803/ai-vs-real-image-classifier" target="_blank">GitHub ↗</a></div>
</div>
""", unsafe_allow_html=True)
