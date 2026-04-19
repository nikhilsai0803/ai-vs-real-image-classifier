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

st.set_page_config(
    page_title="AI vs Real Classifier",
    page_icon="🧿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Session state for page ───────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Detector"

# ─── Constants ────────────────────────────────────────────────
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

# ─── Helpers ──────────────────────────────────────────────────
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

# ─── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:      #06060f;
  --surface: #0b0b18;
  --card:    #10101e;
  --card2:   #141425;
  --border:  #1c1c30;
  --border2: #252540;
  --text:    #eeeef8;
  --muted:   #4e4e6a;
  --dim:     #7878a0;
  --accent:  #00ffd0;
  --purple:  #a06fff;
  --red:     #ff4d6d;
  --green:   #00e5a0;
  --amber:   #ffb830;
  --blue:    #4da6ff;
  --head:    'Syne', sans-serif;
  --mono:    'JetBrains Mono', monospace;
  --margin:  2.2rem;
}

/* ── Streamlit resets ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton, .stToolbar { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.stApp { background: var(--bg) !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* ══════════════════════════════════
   TOP NAVBAR
══════════════════════════════════ */
.navbar {
  position: sticky; top: 0; z-index: 1000;
  background: rgba(6,6,15,0.92);
  backdrop-filter: blur(20px) saturate(180%);
  -webkit-backdrop-filter: blur(20px) saturate(180%);
  border-bottom: 1px solid var(--border);
  padding: 0 var(--margin);
  display: flex; align-items: center; gap: 2rem;
  height: 60px; position: relative;
}
.navbar::after {
  content: '';
  position: absolute; bottom: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(0,255,208,0.25), transparent);
}
.nav-logo {
  font-family: var(--head);
  font-size: 1.15rem; font-weight: 800;
  color: var(--text); letter-spacing: -0.03em;
  white-space: nowrap; flex-shrink: 0;
  display: flex; align-items: center; gap: 0.5rem;
}
.nav-logo em { color: var(--accent); font-style: normal; }
.nav-logo-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 10px var(--accent);
  animation: pulse 2s ease-in-out infinite;
  flex-shrink: 0;
}
@keyframes pulse {
  0%,100% { box-shadow: 0 0 6px var(--accent); }
  50% { box-shadow: 0 0 18px var(--accent), 0 0 32px rgba(0,255,208,0.3); }
}
.nav-right {
  margin-left: auto; display: flex; align-items: center;
  gap: 0.8rem; flex-shrink: 0;
}
.nav-badge {
  font-family: var(--mono); font-size: 0.52rem; letter-spacing: 0.14em;
  color: var(--muted); border: 1px solid var(--border2);
  border-radius: 3px; padding: 0.2rem 0.55rem; text-transform: uppercase;
}
.nav-gh {
  font-family: var(--mono); font-size: 0.62rem; letter-spacing: 0.06em;
  color: var(--accent); border: 1px solid rgba(0,255,208,0.25);
  border-radius: 4px; padding: 0.35rem 0.8rem;
  text-decoration: none; transition: all 0.18s;
  background: rgba(0,255,208,0.04);
}
.nav-gh:hover { background: rgba(0,255,208,0.12); border-color: rgba(0,255,208,0.5); }

/* ══════════════════════════════════
   PAGE WRAPPER
══════════════════════════════════ */
.page-wrap {
  padding: 0 var(--margin);
  max-width: 1400px;
  margin: 0 auto;
}
.page-wrap-full { padding: 0 var(--margin); }

/* ══════════════════════════════════
   HERO
══════════════════════════════════ */
.hero {
  padding: 3.5rem 0 2.8rem;
  border-bottom: 1px solid var(--border);
  position: relative; overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute; top: -120px; right: -60px;
  width: 500px; height: 500px;
  background: radial-gradient(circle, rgba(0,255,208,0.05) 0%, transparent 60%);
  pointer-events: none;
}
.hero::after {
  content: '';
  position: absolute; bottom: -80px; left: 30%;
  width: 350px; height: 350px;
  background: radial-gradient(circle, rgba(160,111,255,0.04) 0%, transparent 60%);
  pointer-events: none;
}
.hero-eyebrow {
  font-family: var(--mono); font-size: 0.58rem;
  letter-spacing: 0.3em; color: var(--accent);
  text-transform: uppercase; margin-bottom: 0.7rem;
  display: flex; align-items: center; gap: 0.6rem;
}
.hero-eyebrow::before {
  content: ''; display: inline-block;
  width: 24px; height: 1px; background: var(--accent);
}
.hero-h1 {
  font-family: var(--head);
  font-size: clamp(2.2rem, 4.5vw, 3.5rem);
  font-weight: 800; color: var(--text);
  letter-spacing: -0.04em; line-height: 1.02;
  margin-bottom: 0.8rem;
}
.hero-h1 b { color: var(--accent); }
.hero-p {
  font-family: var(--mono); font-size: 0.75rem;
  color: var(--dim); line-height: 1.9; max-width: 560px;
}

/* ══════════════════════════════════
   CONTENT SECTION
══════════════════════════════════ */
.content { padding: 2.5rem 0; }

/* ── Cards ── */
.card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 8px; padding: 1.5rem 1.6rem; margin-bottom: 1rem;
  transition: border-color 0.2s;
}
.card:hover { border-color: var(--border2); }
.card.a { border-top: 2px solid var(--accent); }
.card.p { border-top: 2px solid var(--purple); }
.card.g { border-top: 2px solid var(--green); }
.card.r { border-top: 2px solid var(--red); }
.card.am{ border-top: 2px solid var(--amber); }
.card.b { border-top: 2px solid var(--blue); }
.card-lbl {
  font-family: var(--mono); font-size: 0.52rem;
  letter-spacing: 0.24em; color: var(--muted);
  text-transform: uppercase; margin-bottom: 0.6rem;
}
.card-h { font-family: var(--head); font-size: 1rem; font-weight: 700; color: var(--text); margin-bottom: 0.5rem; }
.card-p { font-family: var(--mono); font-size: 0.71rem; color: var(--dim); line-height: 1.9; }

/* ── Stats ── */
.stats {
  display: grid; grid-template-columns: repeat(4,1fr);
  gap: 1rem; margin: 2rem 0;
}
.stat {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 8px; padding: 1.4rem; text-align: center;
  position: relative; overflow: hidden;
  transition: border-color 0.2s, transform 0.2s;
}
.stat:hover { border-color: var(--border2); transform: translateY(-2px); }
.stat::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, var(--border2), transparent);
}
.stat-v { font-family: var(--head); font-size: 2.2rem; font-weight: 800; line-height: 1; display: block; margin-bottom: 0.4rem; }
.stat-l { font-family: var(--mono); font-size: 0.52rem; letter-spacing: 0.18em; color: var(--muted); text-transform: uppercase; }

/* ── Tags ── */
.tags { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-top: 0.9rem; }
.tag { font-family: var(--mono); font-size: 0.54rem; letter-spacing: 0.1em; text-transform: uppercase; border-radius: 3px; padding: 0.2rem 0.6rem; border: 1px solid; }
.ta { color: var(--accent); border-color: rgba(0,255,208,0.3); background: rgba(0,255,208,0.06); }
.tp { color: var(--purple); border-color: rgba(160,111,255,0.3); background: rgba(160,111,255,0.06); }
.tg { color: var(--green);  border-color: rgba(0,229,160,0.3); background: rgba(0,229,160,0.06); }
.tam{ color: var(--amber);  border-color: rgba(255,184,48,0.3); background: rgba(255,184,48,0.06); }
.tb { color: var(--blue);   border-color: rgba(77,166,255,0.3); background: rgba(77,166,255,0.06); }

/* ── Section heading ── */
.sec {
  font-family: var(--head); font-size: 1.15rem; font-weight: 700;
  color: var(--text); letter-spacing: -0.02em;
  margin-bottom: 1.2rem; margin-top: 2rem;
  display: flex; align-items: center; gap: 0.8rem;
}
.sec::after { content: ''; flex: 1; height: 1px; background: var(--border); }
.hr { height: 1px; background: var(--border); margin: 2.5rem 0; }

/* ══════════════════════════════════
   DETECTOR PANELS
══════════════════════════════════ */
.det-panel {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 10px; padding: 2rem;
}
.det-panel.right { background: var(--surface); }
.det-panel-lbl {
  font-family: var(--mono); font-size: 0.52rem;
  letter-spacing: 0.24em; color: var(--accent);
  text-transform: uppercase; margin-bottom: 1.2rem;
  padding-bottom: 0.8rem; border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 0.5rem;
}
.det-panel-lbl span { color: var(--muted); }

.model-pill {
  display: flex; align-items: center; gap: 1rem;
  border: 1px solid var(--border2); border-radius: 6px;
  background: var(--card2); padding: 0.9rem 1.1rem; margin-bottom: 0.8rem;
}
.mdot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--accent); box-shadow: 0 0 8px var(--accent);
  animation: pulse 2s ease-in-out infinite; flex-shrink: 0;
}
.model-nm { font-family: var(--head); font-size: 0.9rem; font-weight: 700; color: var(--text); flex: 1; }
.mpill {
  font-family: var(--mono); font-size: 0.5rem; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--accent);
  border: 1px solid rgba(0,255,208,0.25); border-radius: 3px;
  padding: 0.18rem 0.5rem; background: rgba(0,255,208,0.05);
}
.model-desc { font-family: var(--mono); font-size: 0.68rem; color: var(--muted); line-height: 1.75; margin-bottom: 1.5rem; }

.img-wrap { border: 1px solid var(--border2); border-radius: 7px; overflow: hidden; background: var(--bg); margin: 0.8rem 0; }
.img-meta {
  font-family: var(--mono); font-size: 0.57rem; letter-spacing: 0.1em;
  color: var(--muted); text-transform: uppercase; padding: 0.5rem 0.9rem;
  border-top: 1px solid var(--border); display: flex; gap: 1rem;
}

/* result card */
.result {
  border: 1px solid var(--border2); border-radius: 8px;
  background: var(--card2); padding: 1.8rem;
  position: relative; overflow: hidden;
  animation: slideUp 0.35s ease;
}
@keyframes slideUp {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}
.result::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; }
.result.real::before  { background: linear-gradient(90deg, transparent, var(--green), transparent); }
.result.fake::before  { background: linear-gradient(90deg, transparent, var(--red), transparent); }
.result.unsure::before{ background: linear-gradient(90deg, transparent, var(--amber), transparent); }

.verdict-lbl { font-family: var(--mono); font-size: 0.5rem; letter-spacing: 0.24em; color: var(--muted); text-transform: uppercase; margin-bottom: 0.2rem; }
.verdict { font-family: var(--head); font-size: 2.6rem; font-weight: 800; letter-spacing: -0.04em; line-height: 1.0; margin: 0.3rem 0; }
.verdict.real   { color: var(--green); }
.verdict.fake   { color: var(--red); }
.verdict.unsure { color: var(--amber); }
.verdict-conf { font-family: var(--mono); font-size: 0.68rem; color: var(--muted); margin-bottom: 1.5rem; }

.brow { display: flex; justify-content: space-between; font-family: var(--mono); font-size: 0.6rem; color: var(--muted); margin-bottom: 0.35rem; }
.btrack { height: 4px; background: var(--border); border-radius: 4px; overflow: hidden; margin-bottom: 0.75rem; }
.bfill { height: 100%; border-radius: 4px; transition: width 0.6s cubic-bezier(0.4,0,0.2,1); }
.bgreen { background: linear-gradient(90deg, var(--green), #00ffb3); }
.bred   { background: linear-gradient(90deg, var(--red), #ff8099); }

.scores { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.6rem; margin-top: 1rem; }
.sc { background: var(--surface); border: 1px solid var(--border); border-radius: 5px; padding: 0.7rem; text-align: center; }
.sc-v { font-family: var(--mono); font-size: 0.95rem; font-weight: 500; color: var(--text); display: block; }
.sc-l { font-family: var(--mono); font-size: 0.49rem; letter-spacing: 0.15em; color: var(--muted); text-transform: uppercase; display: block; margin-top: 0.25rem; }

.warn { display: inline-flex; align-items: center; gap: 0.45rem; font-family: var(--mono); font-size: 0.6rem; color: var(--amber); border: 1px solid rgba(255,184,48,0.3); background: rgba(255,184,48,0.06); border-radius: 4px; padding: 0.3rem 0.7rem; margin-bottom: 1rem; }

.empty { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 380px; text-align: center; gap: 1rem; }
.empty-icon { font-size: 3.5rem; opacity: 0.12; }
.empty-t { font-family: var(--mono); font-size: 0.7rem; letter-spacing: 0.2em; color: var(--muted); text-transform: uppercase; }
.empty-s { font-family: var(--mono); font-size: 0.62rem; color: var(--muted); max-width: 230px; line-height: 1.9; }

/* ── Steps ── */
.step {
  display: flex; gap: 1.4rem; padding: 1.4rem 1.5rem; background: var(--card);
  border: 1px solid var(--border); border-radius: 8px;
  margin-bottom: 0.9rem; align-items: flex-start; transition: border-color 0.2s;
}
.step:hover { border-color: var(--border2); }
.step-n { font-family: var(--head); font-size: 2rem; font-weight: 800; color: var(--border2); line-height: 1; min-width: 2.4rem; text-align: center; padding-top: 0.1rem; }
.step-h { font-family: var(--head); font-size: 0.95rem; font-weight: 700; color: var(--text); margin-bottom: 0.4rem; }
.step-b { font-family: var(--mono); font-size: 0.7rem; color: var(--dim); line-height: 1.9; }

/* ── Code ── */
.code {
  background: #050510; border: 1px solid var(--border);
  border-left: 3px solid var(--accent); border-radius: 6px;
  padding: 1.3rem 1.5rem; font-family: var(--mono); font-size: 0.68rem;
  color: #8080b0; line-height: 2.1; overflow-x: auto; white-space: pre; margin: 1rem 0;
}
.kw { color: var(--purple); } .fn { color: var(--accent); }
.st { color: var(--green); }  .cm { color: #3a3a5c; } .nm { color: var(--amber); }

/* ── Table ── */
.tbl { width: 100%; border-collapse: collapse; }
.tbl td { font-family: var(--mono); font-size: 0.69rem; padding: 0.65rem 0.7rem; border-bottom: 1px solid var(--border); vertical-align: top; }
.tbl td:first-child { color: var(--muted); width: 38%; font-size: 0.59rem; letter-spacing: 0.07em; text-transform: uppercase; }
.tbl td:last-child { color: var(--text); }

/* ── Footer ── */
.foot {
  border-top: 1px solid var(--border); padding: 1.2rem var(--margin);
  display: flex; justify-content: space-between; align-items: center;
  background: var(--surface); margin-top: 3rem;
}
.foot-l { font-family: var(--mono); font-size: 0.56rem; letter-spacing: 0.14em; color: var(--muted); text-transform: uppercase; }
.foot-r { font-family: var(--mono); font-size: 0.56rem; color: var(--muted); }
.foot-r a { color: var(--accent); text-decoration: none; }

/* ── Streamlit nav button overrides ── */
div[data-testid="stHorizontalBlock"] button {
  background: transparent !important; border: 1px solid transparent !important;
  color: var(--muted) !important; border-radius: 4px !important;
  transition: all 0.18s !important; font-family: var(--mono) !important;
}
div[data-testid="stHorizontalBlock"] button:hover {
  color: var(--text) !important; background: var(--card) !important;
  border-color: var(--border2) !important;
}
div[data-testid="stHorizontalBlock"] button p {
  font-family: var(--mono) !important; font-size: 0.68rem !important;
  letter-spacing: 0.06em !important; text-transform: uppercase !important;
}

/* ── Other widget overrides ── */
.stSelectbox > div > div {
  background: var(--card2) !important; border: 1px solid var(--border2) !important;
  border-radius: 5px !important; color: var(--text) !important;
  font-family: var(--mono) !important; font-size: 0.82rem !important;
}
.stSelectbox label { color: var(--muted) !important; font-family: var(--mono) !important; font-size: 0.58rem !important; letter-spacing: 0.16em !important; text-transform: uppercase !important; }
.stFileUploader > div { border: 1px dashed var(--border2) !important; border-radius: 7px !important; background: var(--card2) !important; }
.stFileUploader > div:hover { border-color: var(--accent) !important; }
.stFileUploader label { color: var(--muted) !important; font-family: var(--mono) !important; font-size: 0.58rem !important; letter-spacing: 0.16em !important; text-transform: uppercase !important; }
.stImage img { border-radius: 0 !important; display: block !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# ─── TOP NAVBAR ───────────────────────────────────────────────
pages = ["🧿 Detector", "📖 About", "⚙️ Tech Stack", "🔬 How It Works"]

st.markdown(f"""
<div style="position:sticky;top:0;z-index:1000;background:rgba(6,6,15,0.94);
  backdrop-filter:blur(20px);border-bottom:1px solid #1c1c30;
  padding:0 2.2rem;display:flex;align-items:center;gap:1.5rem;
  height:60px;overflow:hidden;">
  <div class="nav-logo">
    <div class="nav-logo-dot"></div>
    AI<em>vs</em>Real
  </div>
  <div style="width:1px;height:22px;background:#1c1c30;flex-shrink:0;"></div>
  <div style="flex:1"></div>
  <span class="nav-badge">TF 2.21</span>
  <a class="nav-gh" href="https://github.com/nikhilsai0803/ai-vs-real-image-classifier" target="_blank">GitHub ↗</a>
</div>
""", unsafe_allow_html=True)

# Nav buttons row
nav_cols = st.columns([2, 1, 1, 1, 1, 2])
labels = ["Detector", "About", "Tech Stack", "How It Works"]
col_indices = [1, 2, 3, 4]
for i, label in enumerate(labels):
    full_page = pages[i]
    with nav_cols[col_indices[i]]:
        if st.button(label, key=f"nav_{label}", use_container_width=True):
            st.session_state.page = full_page
            st.rerun()

# Active indicator via CSS targeting button text
active_label = st.session_state.page.split(" ", 1)[1] if " " in st.session_state.page else st.session_state.page
st.markdown(f"""
<style>
/* Highlight the active nav button by targeting its text content approach */
div[data-testid="stHorizontalBlock"] button[kind="secondary"] {{
  background: transparent !important;
}}
</style>
""", unsafe_allow_html=True)

page = st.session_state.page

# ─── HERO ─────────────────────────────────────────────────────
HEROES = {
    "Detector": ("Live Inference Engine", "Image <b>Detector</b>", "Upload any image. The selected model classifies it as AI-generated or a real photograph with full confidence breakdown."),
    "About":    ("Project Overview", "About <b>This Project</b>", "A deep learning binary classifier distinguishing AI-generated artwork from real photographs using transfer learning on three pretrained architectures."),
    "Tech":     ("Tools & Libraries", "Tech <b>Stack</b>", "Every library, framework, and tool used to build, train, evaluate, and deploy this project."),
    "How":      ("Technical Deep Dive", "How It <b>Works</b>", "The full pipeline — from raw images on disk to a confident prediction — explained step by step."),
}
for key, (eye, h1, sub) in HEROES.items():
    if key in page:
        st.markdown(f"""
        <div class="page-wrap-full">
          <div class="hero">
            <div class="hero-eyebrow">{eye}</div>
            <div class="hero-h1">{h1}</div>
            <div class="hero-p">{sub}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        break

# ══════════════════════════════════════════════════════════════
# PAGE 1 — DETECTOR
# ══════════════════════════════════════════════════════════════
if "Detector" in page:
    st.markdown('<div class="page-wrap"><div style="padding: 2rem 0;">', unsafe_allow_html=True)
    col_l, col_r = st.columns([1, 1], gap="medium")

    with col_l:
        st.markdown('<div class="det-panel">', unsafe_allow_html=True)
        st.markdown('<div class="det-panel-lbl">01 · Select Model <span>— choose architecture</span></div>', unsafe_allow_html=True)

        selected = st.selectbox("model", list(MODEL_PATHS.keys()), label_visibility="collapsed")
        info = MODEL_INFO[selected]
        st.markdown(f"""
        <div class="model-pill">
          <div class="mdot"></div>
          <div class="model-nm">{selected}</div>
          <span class="mpill">Fine-tuned</span>
          <span class="mpill">ImageNet</span>
        </div>
        <div class="model-desc">{info['desc']}&nbsp;&nbsp;<span style="color:var(--text)">{info['params']} params · {info['speed']}</span></div>
        """, unsafe_allow_html=True)

        model = load_model(selected)
        if model is None:
            st.error(f"**Model not found:** `{MODEL_PATHS[selected]}`\n\nRun the training notebook to generate `classifier_outputs/`.")
            st.stop()

        st.markdown('<div class="det-panel-lbl" style="margin-top:0.5rem;">02 · Upload Image <span>— jpg, png, webp</span></div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")

        if uploaded:
            pil_img = Image.open(io.BytesIO(uploaded.read()))
            w, h = pil_img.size
            st.markdown('<div class="img-wrap">', unsafe_allow_html=True)
            st.image(pil_img, use_container_width=True)
            st.markdown(f'<div class="img-meta"><span>{uploaded.name}</span><span>{w}×{h}px</span><span>{pil_img.mode}</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="det-panel right">', unsafe_allow_html=True)
        st.markdown('<div class="det-panel-lbl">03 · Result <span>— classification output</span></div>', unsafe_allow_html=True)

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

            warn_html = '<div class="warn">⚠ LOW CONFIDENCE — result may be unreliable</div>' if uncertain else ""

            st.markdown(f"""
            <div class="result {cls}">
              <div class="verdict-lbl">Verdict</div>
              <div style="font-size:2.2rem;margin:0.3rem 0;">{icon}</div>
              <div class="verdict {vc}">{label}</div>
              <div class="verdict-conf">Confidence: {conf}% &nbsp;·&nbsp; Raw score: {raw:.4f}</div>
              {warn_html}
              <div class="brow"><span>🤖 AI-Generated</span><span>{ai_pct}%</span></div>
              <div class="btrack"><div class="bfill bred" style="width:{ai_pct}%"></div></div>
              <div class="brow"><span>📷 Real Photo</span><span>{rpct}%</span></div>
              <div class="btrack"><div class="bfill bgreen" style="width:{rpct}%"></div></div>
              <div class="scores">
                <div class="sc"><span class="sc-v">{raw:.4f}</span><span class="sc-l">Raw Score</span></div>
                <div class="sc"><span class="sc-v">{conf}%</span><span class="sc-l">Confidence</span></div>
                <div class="sc"><span class="sc-v" style="font-size:0.62rem;">{selected}</span><span class="sc-l">Model</span></div>
              </div>
            </div>
            <div class="card" style="margin-top:0.9rem;">
              <div class="card-lbl">How to read this</div>
              <div class="card-p">
                Score &gt; 0.5 → <span style="color:var(--green)">Real Photo</span> &nbsp;|&nbsp;
                Score &lt; 0.5 → <span style="color:var(--red)">AI-Generated</span><br>
                Confidence below <strong style="color:var(--amber)">85%</strong> is flagged as uncertain.
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty">
              <div class="empty-icon">🧿</div>
              <div class="empty-t">Awaiting Image</div>
              <div class="empty-s">Upload a JPG, PNG, or WEBP in the left panel to run the classifier</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — ABOUT
# ══════════════════════════════════════════════════════════════
elif "About" in page:
    st.markdown('<div class="page-wrap"><div class="content">', unsafe_allow_html=True)

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
              <tr><td>Validation</td><td>Corrupt images removed before training</td></tr>
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

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec">Project Goals</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown("""<div class="card"><div style="font-size:1.8rem;margin-bottom:0.6rem;">🎯</div>
        <div class="card-h">Accurate Classification</div>
        <div class="card-p">Learn genuine visual features — not brightness or colour shortcuts — to reliably separate AI from real.</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card"><div style="font-size:1.8rem;margin-bottom:0.6rem;">⚡</div>
        <div class="card-h">Lightweight & Fast</div>
        <div class="card-p">Mobile-scale architectures so inference runs in seconds even on CPU — no GPU required.</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="card"><div style="font-size:1.8rem;margin-bottom:0.6rem;">🌐</div>
        <div class="card-h">Deployable UI</div>
        <div class="card-p">A polished web interface anyone can use — no code required. Upload, click, get your answer.</div></div>""", unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 3 — TECH STACK
# ══════════════════════════════════════════════════════════════
elif "Tech" in page:
    st.markdown('<div class="page-wrap"><div class="content">', unsafe_allow_html=True)
    st.markdown('<div class="sec">Core ML Framework</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown("""<div class="card a"><div style="font-size:1.5rem;margin-bottom:0.5rem;">🧠</div>
        <div class="card-h">TensorFlow 2.21</div>
        <div class="card-p">Full ML lifecycle — <code style="color:var(--accent)">tf.data</code> pipelines, training loops, model saving and serving. Open-source by Google.</div>
        <div class="tags"><span class="tag ta">Core</span><span class="tag tp">Google</span></div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card p"><div style="font-size:1.5rem;margin-bottom:0.5rem;">🔷</div>
        <div class="card-h">Keras</div>
        <div class="card-p">High-level API inside TensorFlow. Defines model architecture, compiles, runs training. The <code style="color:var(--accent)">applications</code> module provides all three pretrained bases.</div>
        <div class="tags"><span class="tag tp">API</span><span class="tag ta">Built-in</span></div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="card g"><div style="font-size:1.5rem;margin-bottom:0.5rem;">📦</div>
        <div class="card-h">NumPy</div>
        <div class="card-p">Converts PIL images to float32 arrays, stacks batches, post-processes raw sigmoid scores before display in the UI.</div>
        <div class="tags"><span class="tag tg">Numerical</span></div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec">Pretrained Architectures</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown("""<div class="card b"><div style="font-size:1.3rem;margin-bottom:0.5rem;">📱</div>
        <div class="card-h">MobileNetV2</div>
        <div class="card-p"><table class="tbl">
          <tr><td>Params</td><td>2.4M total</td></tr>
          <tr><td>Input</td><td>[0,255] → [-1, 1]</td></tr>
          <tr><td>Design</td><td>Depthwise separable + inverted residuals</td></tr>
          <tr><td>Best for</td><td>Real-time, mobile inference</td></tr>
        </table></div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card a"><div style="font-size:1.3rem;margin-bottom:0.5rem;">⚖️</div>
        <div class="card-h">EfficientNetB0</div>
        <div class="card-p"><table class="tbl">
          <tr><td>Params</td><td>4.2M total</td></tr>
          <tr><td>Input</td><td>[0,255] → normalised internally</td></tr>
          <tr><td>Design</td><td>Compound scaling of depth, width, resolution</td></tr>
          <tr><td>Best for</td><td>Highest accuracy per parameter</td></tr>
        </table></div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="card p"><div style="font-size:1.3rem;margin-bottom:0.5rem;">🔬</div>
        <div class="card-h">NASNetMobile</div>
        <div class="card-p"><table class="tbl">
          <tr><td>Params</td><td>4.4M total</td></tr>
          <tr><td>Input</td><td>[0,255] → [-1, 1]</td></tr>
          <tr><td>Design</td><td>Neural Architecture Search by Google</td></tr>
          <tr><td>Best for</td><td>Robustness, generalisation</td></tr>
        </table></div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec">Supporting Libraries & Deployment</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("""<div class="card"><div class="card-lbl">Data & Visualisation</div>
        <table class="tbl">
          <tr><td>scikit-learn</td><td>train_test_split, classification_report, confusion_matrix</td></tr>
          <tr><td>Matplotlib</td><td>Training plots — curves, distributions, sample grids</td></tr>
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

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec">requirements.txt</div>', unsafe_allow_html=True)
    st.markdown("""<div class="code"><span class="cm"># pip install -r requirements.txt</span>

<span class="kw">tensorflow</span>==<span class="nm">2.21.0</span>
<span class="kw">streamlit</span>&gt;=<span class="nm">1.35.0</span>
<span class="kw">numpy</span>&gt;=<span class="nm">1.24.0</span>
<span class="kw">Pillow</span>&gt;=<span class="nm">10.0.0</span>
<span class="kw">scikit-learn</span>&gt;=<span class="nm">1.3.0</span>
<span class="kw">matplotlib</span>&gt;=<span class="nm">3.7.0</span>
<span class="kw">seaborn</span>&gt;=<span class="nm">0.12.0</span></div>""", unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 4 — HOW IT WORKS
# ══════════════════════════════════════════════════════════════
elif "How" in page:
    st.markdown('<div class="page-wrap"><div class="content">', unsafe_allow_html=True)
    st.markdown('<div class="sec">Training Pipeline</div>', unsafe_allow_html=True)

    steps = [
        ("Image Collection & Validation",
         "All images are scanned recursively from <code style='color:var(--accent)'>AiArtData/</code> and <code style='color:var(--accent)'>RealArt/</code>. Before training begins, every file is decoded by TensorFlow. Corrupt files, truncated JPEGs, and images smaller than 10×10px are <strong style='color:var(--red)'>detected and removed</strong> so they never cause silent failures mid-training."),
        ("Data Split & tf.data Pipeline",
         "Images are shuffled and split 70/15/15 (train/val/test). A <code style='color:var(--accent)'>tf.data.Dataset</code> pipeline loads images lazily on demand. Training data gets light augmentation: random flip, ±10% brightness, ±15% contrast. Critical fix: images stay as <code style='color:var(--accent)'>float32 in [0, 255]</code> — <strong style='color:var(--red)'>no /255 division</strong>. Each model's preprocess_input handles its own scaling."),
        ("Phase 1 — Head Training",
         "The base model is <strong>fully frozen</strong>. Only the classification head trains: GlobalAveragePooling → BatchNorm → Dense(256) → Dropout(0.4) → Dense(64) → Dropout(0.2) → Sigmoid. Learning rate: 1e-3. EarlyStopping monitors val_loss with patience=3. Typically converges in 5–9 epochs, reaching ~70–75% validation accuracy."),
        ("Phase 2 — Surgical Fine-tuning",
         "Only the <strong>last 30 layers</strong> of the base model are unfrozen — not the whole network, which would destroy ImageNet weights. Learning rate drops to 1e-5. A <strong>fresh EarlyStopping callback</strong> is created (the Phase 1 callback had stale state). This phase teaches the model AI-art-specific patterns: GAN noise, synthetic gradients, unnatural smoothness."),
        ("Evaluation & Saving",
         "All models are evaluated on the held-out test set. Confusion matrices, precision/recall/F1, and accuracy are reported. Every graph and every model is saved to <code style='color:var(--accent)'>classifier_outputs/</code> — never to the base directory. Models saved as <code style='color:var(--accent)'>.keras</code> files for serving in this app."),
    ]

    for i, (title, body) in enumerate(steps, 1):
        st.markdown(f"""
        <div class="step">
          <div class="step-n">0{i}</div>
          <div><div class="step-h">{title}</div><div class="step-b">{body}</div></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec">Inference Pipeline</div>', unsafe_allow_html=True)

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
          6. Score ≥ 0.5 → <span style="color:var(--green)">Real</span> &nbsp;|&nbsp; Score &lt; 0.5 → <span style="color:var(--red)">AI</span><br>
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
          Dividing by 255 first sent wrong values through the scaler — producing garbage activations. All models predicted only "Real" at <strong style="color:var(--red)">49% accuracy</strong> (random guessing).
        </div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec">Key Code Fixes</div>', unsafe_allow_html=True)
    st.markdown("""<div class="code"><span class="cm"># ❌ WRONG — caused 49% accuracy (all models predicted only "Real")</span>
<span class="kw">def</span> <span class="fn">parse_image</span>(path, label):
    img = tf.<span class="fn">cast</span>(img, tf.float32) / <span class="nm">255.0</span>  <span class="cm"># ← breaks preprocess_input</span>
    <span class="kw">return</span> img, label

<span class="cm"># ✅ CORRECT — cast only, let each model handle its own scaling</span>
<span class="kw">def</span> <span class="fn">parse_image</span>(path, label):
    img = tf.<span class="fn">cast</span>(img, tf.float32)  <span class="cm"># NO /255</span>
    <span class="kw">return</span> img, label

<span class="cm"># ❌ WRONG fine-tuning — unfreezes ALL layers, destroys ImageNet weights</span>
model.layers[<span class="nm">1</span>].trainable = <span class="kw">True</span>

<span class="cm"># ✅ CORRECT — only last 30 layers + fresh EarlyStopping for Phase 2</span>
<span class="kw">for</span> layer <span class="kw">in</span> base.layers:        layer.trainable = <span class="kw">False</span>
<span class="kw">for</span> layer <span class="kw">in</span> base.layers[-<span class="nm">30</span>:]:  layer.trainable = <span class="kw">True</span>
cb_p2 = [keras.callbacks.<span class="fn">EarlyStopping</span>(monitor=<span class="st">'val_loss'</span>, patience=<span class="nm">4</span>)]</div>""", unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────
st.markdown("""
<div class="foot">
  <div class="foot-l">AI vs Real Image Classifier &nbsp;·&nbsp; TensorFlow &nbsp;·&nbsp; Transfer Learning &nbsp;·&nbsp; Streamlit</div>
  <div class="foot-r"><a href="https://github.com/nikhilsai0803/ai-vs-real-image-classifier" target="_blank">GitHub ↗</a></div>
</div>
""", unsafe_allow_html=True)
