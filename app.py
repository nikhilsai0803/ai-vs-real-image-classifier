"""
app.py — AI vs Real Image Classifier · Fixed Single Page + No Empty Boxes
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

if "page" not in st.session_state:
    st.session_state.page = "Detector"

# ── NAV PAGES ─────────────────────────────────────────────────────
NAV_PAGES = ["Detector", "About", "Tech Stack", "How It Works"]

# ── Handle navigation via query params (Single Page Navigation) ──
qp = st.query_params
if "p" in qp:
    requested = qp["p"].replace("+", " ")
    if requested in NAV_PAGES:
        st.session_state.page = requested
        st.query_params.clear()
        st.rerun()

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

page = st.session_state.page

# ── CSS (unchanged) ──────────────────────────────────────────────────────
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

/* Streamlit resets */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton, .stToolbar { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.stApp { background: var(--bg) !important; }
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"]  { display: none !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }

/* Navbar */
.topnav {
  position: sticky; top: 0; z-index: 999;
  height: 58px;
  display: flex; align-items: center;
  padding: 0 var(--side);
  gap: 2.4rem;
  background: rgba(7,7,15,0.90);
  backdrop-filter: blur(18px) saturate(160%);
  border-bottom: 1px solid var(--border);
}
.topnav::after {
  content: '';
  position: absolute; bottom: -1px; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent 0%, rgba(0,255,208,.18) 50%, transparent 100%);
}

.tnav-logo {
  display: flex; align-items: center; gap: 0.5rem; flex-shrink: 0;
  font-family: var(--head); font-size: 1.12rem; font-weight: 800;
  color: var(--text); letter-spacing: -0.03em;
}
.tnav-logo em { color: var(--accent); font-style: normal; }
.tnav-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--accent); box-shadow: 0 0 10px var(--accent);
  animation: glow 2.4s ease-in-out infinite;
}
@keyframes glow {
  0%,100% { box-shadow: 0 0 5px var(--accent); }
  50% { box-shadow: 0 0 16px var(--accent), 0 0 28px rgba(0,255,208,.25); }
}

.tnav-div { width: 1px; height: 20px; background: var(--border2); flex-shrink: 0; }

.tnav-links { display: flex; align-items: center; gap: 0.2rem; flex: 1; }
.tnav-link {
  font-family: var(--mono); font-size: 0.67rem;
  letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--muted); text-decoration: none;
  border: 1px solid transparent; border-radius: 4px;
  padding: 0.38rem 0.95rem;
  transition: all .16s;
  cursor: pointer; white-space: nowrap;
}
.tnav-link:hover {
  color: var(--text); background: var(--card); border-color: var(--border2);
}
.tnav-link.active {
  color: var(--accent);
  background: rgba(0,255,208,.07);
  border-color: rgba(0,255,208,.22);
}

.tnav-right {
  margin-left: auto; display: flex; align-items: center;
  gap: 0.7rem; flex-shrink: 0;
}
.tnav-badge {
  font-family: var(--mono); font-size: 0.5rem; letter-spacing: 0.16em;
  color: var(--muted); border: 1px solid var(--border2);
  border-radius: 3px; padding: 0.2rem 0.5rem; text-transform: uppercase;
}
.tnav-gh {
  font-family: var(--mono); font-size: 0.6rem; letter-spacing: 0.06em;
  color: var(--accent); border: 1px solid rgba(0,255,208,.24);
  border-radius: 4px; padding: 0.32rem 0.75rem;
  text-decoration: none; background: rgba(0,255,208,.04);
}
.tnav-gh:hover { background: rgba(0,255,208,.12); border-color: rgba(0,255,208,.5); }

/* Wrappers */
.pw  { padding: 0 var(--side); max-width: 1440px; margin: 0 auto; }
.pfw { padding: 0 var(--side); }

/* Hero */
.hero {
  padding: 2.8rem 0 2.4rem;
  border-bottom: 1px solid var(--border);
  position: relative; overflow: hidden;
}
.hero-eye {
  font-family: var(--mono); font-size: 0.56rem;
  letter-spacing: 0.3em; color: var(--accent);
  text-transform: uppercase; margin-bottom: 0.65rem;
  display: flex; align-items: center; gap: 0.55rem;
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

/* Detector Panels */
.det-panel {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 10px; padding: 1.8rem;
}
.det-panel.right { background: var(--surf); }
.dp-label {
  font-family: var(--mono); font-size: 0.5rem;
  letter-spacing: 0.24em; text-transform: uppercase;
  color: var(--accent); padding-bottom: 0.8rem;
  border-bottom: 1px solid var(--border); margin-bottom: 1.1rem;
  display: flex; gap: 0.5rem;
}
.dp-label span { color: var(--muted); }

.mpill-row {
  display: flex; align-items: center; gap: 0.9rem;
  background: var(--card2); border: 1px solid var(--border2);
  border-radius: 6px; padding: 0.82rem 1rem; margin-bottom: 0.7rem;
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

.img-box { border: 1px solid var(--border2); border-radius: 6px; overflow: hidden; background: var(--bg); margin: 0.6rem 0; }
.img-meta {
  font-family: var(--mono); font-size: 0.54rem;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--muted); padding: 0.45rem 0.8rem;
  border-top: 1px solid var(--border); display: flex; gap: 1rem;
}

.result {
  border: 1px solid var(--border2); border-radius: 8px;
  background: var(--card2); padding: 1.6rem;
  position: relative; overflow: hidden;
  animation: up .3s ease; margin-bottom: 0.9rem;
}
@keyframes up { from { opacity:0; transform:translateY(8px);} to {opacity:1;transform:translateY(0);} }
.result::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; }
.result.real::before  { background: linear-gradient(90deg,transparent,var(--green),transparent); }
.result.fake::before  { background: linear-gradient(90deg,transparent,var(--red),transparent); }
.result.unsure::before{ background: linear-gradient(90deg,transparent,var(--amber),transparent); }

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
.sc { background:var(--surf); border:1px solid var(--border); border-radius:5px; padding:.65rem; text-align:center; }
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
}
.info-card .card-lbl { font-family:var(--mono); font-size:.48rem; letter-spacing:.22em; color:var(--muted); text-transform:uppercase; margin-bottom:.45rem; }
.info-card .card-p   { font-family:var(--mono); font-size:.68rem; color:var(--dim); line-height:1.85; }

.empty {
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  min-height:360px; text-align:center; gap:.8rem;
}
.empty-icon { font-size:3rem; opacity:.1; }
.empty-t { font-family:var(--mono); font-size:.66rem; letter-spacing:.2em; color:var(--muted); text-transform:uppercase; }
.empty-s { font-family:var(--mono); font-size:.6rem; color:var(--muted); max-width:210px; line-height:1.9; }

/* Shared */
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
  transition:border-color .2s;
}
.card:hover { border-color:var(--border2); }
.card.a{border-top:2px solid var(--accent);}
.card.p{border-top:2px solid var(--purple);}
.card.g{border-top:2px solid var(--green);}
.card.r{border-top:2px solid var(--red);}
.card.am{border-top:2px solid var(--amber);}
.card.b{border-top:2px solid var(--blue);}
.card-lbl { font-family:var(--mono); font-size:.5rem; letter-spacing:.24em; color:var(--muted); text-transform:uppercase; margin-bottom:.55rem; }
.card-h   { font-family:var(--head); font-size:.96rem; font-weight:700; color:var(--text); margin-bottom:.45rem; }
.card-p   { font-family:var(--mono); font-size:.7rem; color:var(--dim); line-height:1.9; }

.stats { display:grid; grid-template-columns:repeat(4,1fr); gap:.9rem; margin:1.8rem 0; }
.stat {
  background:var(--card); border:1px solid var(--border); border-radius:8px;
  padding:1.3rem; text-align:center; transition:border-color .2s, transform .2s;
}
.stat:hover { border-color:var(--border2); transform:translateY(-2px); }
.stat-v { font-family:var(--head); font-size:2.1rem; font-weight:800; line-height:1; display:block; margin-bottom:.35rem; }
.stat-l { font-family:var(--mono); font-size:.5rem; letter-spacing:.18em; color:var(--muted); text-transform:uppercase; }

.step {
  display:flex; gap:1.3rem; padding:1.3rem 1.4rem; background:var(--card);
  border:1px solid var(--border); border-radius:8px;
  margin-bottom:.85rem; align-items:flex-start;
}
.step:hover { border-color:var(--border2); }
.step-n { font-family:var(--head); font-size:1.9rem; font-weight:800; color:var(--border2); line-height:1; min-width:2.2rem; text-align:center; padding-top:.05rem; }
.step-h { font-family:var(--head); font-size:.92rem; font-weight:700; color:var(--text); margin-bottom:.35rem; }
.step-b { font-family:var(--mono); font-size:.69rem; color:var(--dim); line-height:1.9; }

.code {
  background:#050510; border:1px solid var(--border);
  border-left:3px solid var(--accent); border-radius:6px;
  padding:1.2rem 1.4rem; font-family:var(--mono); font-size:.67rem;
  color:#8080b0; line-height:2.1; overflow-x:auto; white-space:pre; margin:1rem 0;
}
.kw{color:var(--purple);} .fn{color:var(--accent);} .st{color:var(--green);} .cm{color:#35355a;} .nm{color:var(--amber);}

.tbl { width:100%; border-collapse:collapse; }
.tbl td { font-family:var(--mono); font-size:.68rem; padding:.62rem .7rem; border-bottom:1px solid var(--border); vertical-align:top; }
.tbl td:first-child { color:var(--muted); width:36%; font-size:.57rem; letter-spacing:.07em; text-transform:uppercase; }
.tbl td:last-child  { color:var(--text); }

/* Footer */
.foot {
  border-top:1px solid var(--border); padding:1.1rem var(--side);
  display:flex; justify-content:space-between; align-items:center;
  background:var(--surf); margin-top:3rem;
}
.foot-l { font-family:var(--mono); font-size:.53rem; letter-spacing:.14em; color:var(--muted); text-transform:uppercase; }
.foot-r a { font-family:var(--mono); font-size:.55rem; color:var(--accent); text-decoration:none; }

/* Widget styling */
.stSelectbox > div > div, .stFileUploader > div {
  background:var(--card2) !important; border:1px solid var(--border2) !important;
  border-radius:5px !important; color:var(--text) !important;
}
.stFileUploader > div:hover { border-color:var(--accent) !important; }
.stImage img { border-radius:0 !important; display:block !important; }
</style>
""", unsafe_allow_html=True)

# ── NAVBAR ─────────────────────────────────────────────────────────────
def nav_link_html(label, current):
    cls = "tnav-link active" if label == current else "tnav-link"
    href = f"?p={label.replace(' ', '+')}"
    return f'<a class="{cls}" href="{href}">{label.upper()}</a>'

links_html = "".join(nav_link_html(p, page) for p in NAV_PAGES)

st.markdown(f"""
<nav class="topnav">
  <span class="tnav-logo"><span class="tnav-dot"></span>AI<em>vs</em>Real</span>
  <div class="tnav-div"></div>
  <div class="tnav-links">{links_html}</div>
  <div class="tnav-right">
    <span class="tnav-badge">TF 2.21</span>
    <a class="tnav-gh" href="https://github.com/nikhilsai0803/ai-vs-real-image-classifier" target="_blank">GitHub ↗</a>
  </div>
</nav>
""", unsafe_allow_html=True)

# ── HERO ───────────────────────────────────────────────────────────────
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
# DETECTOR PAGE (Fixed - No empty boxes)
# ════════════════════════════════════════════════════════════
if page == "Detector":
    st.markdown('<div class="pw">', unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="medium")

    with col_l:
        st.markdown('<div class="det-panel">', unsafe_allow_html=True)
        st.markdown('<div class="dp-label">01 · Select Model <span>— architecture</span></div>', unsafe_allow_html=True)

        selected = st.selectbox("Select Model", list(MODEL_PATHS.keys()), label_visibility="collapsed")
        info = MODEL_INFO[selected]
        st.markdown(f"""
        <div class="mpill-row">
          <div class="mpill-dot"></div>
          <div class="mpill-name">{selected}</div>
          <span class="mpill-tag">FINE-TUNED</span>
          <span class="mpill-tag">IMAGENET</span>
        </div>
        <div class="mdesc">{info['desc']}&nbsp;&nbsp;<span style="color:var(--text)">{info['params']} params · {info['speed']}</span></div>
        """, unsafe_allow_html=True)

        model = load_model(selected)
        if model is None:
            st.error(f"**Model not found:** `{MODEL_PATHS[selected]}`")
            st.stop()

        st.markdown('<div class="dp-label" style="margin-top:.6rem;">02 · Upload Image <span>— jpg · png · webp</span></div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")

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
                res = predict(model, tensor)

            label = res["label"]
            conf = res["conf"]
            raw = res["raw"]
            ai_pct = res["ai_pct"]
            rpct = res["real_pct"]
            uncertain = res["uncertain"]

            if uncertain:
                cls, icon, vc = "unsure", "⚠️", "unsure"
            elif "Real" in label:
                cls, icon, vc = "real", "✅", "real"
            else:
                cls, icon, vc = "fake", "🤖", "fake"

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
              <div class="empty-t">AWAITING IMAGE</div>
              <div class="empty-s">Upload a JPG, PNG, or WEBP in the left panel to run inference</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)   # close pw

# ════════════════════════════════════════════════════════════
# OTHER PAGES (About, Tech Stack, How It Works) — Unchanged
# ════════════════════════════════════════════════════════════
else:
    st.markdown('<div class="pw">', unsafe_allow_html=True)

    if page == "About":
        # ... (paste your original About content here - it's unchanged)
        st.markdown("""
        <div class="stats">
          <div class="stat"><span class="stat-v" style="color:var(--accent)">~4.7K</span><span class="stat-l">Training Images</span></div>
          <div class="stat"><span class="stat-v" style="color:var(--purple)">3</span><span class="stat-l">Models Trained</span></div>
          <div class="stat"><span class="stat-v" style="color:var(--green)">2</span><span class="stat-l">Training Phases</span></div>
          <div class="stat"><span class="stat-v" style="color:var(--amber)">85%</span><span class="stat-l">Confidence Threshold</span></div>
        </div>
        """, unsafe_allow_html=True)
        # Add the rest of your About, Tech Stack, How It Works content exactly as before...
        # (I'm keeping it short here for brevity, but copy your original sections)

    elif page == "Tech Stack":
        # Paste your original Tech Stack content
        pass

    elif page == "How It Works":
        # Paste your original How It Works content
        pass

    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div class="foot">
  <div class="foot-l">AI vs Real Image Classifier &nbsp;·&nbsp; TensorFlow &nbsp;·&nbsp; Transfer Learning &nbsp;·&nbsp; Streamlit</div>
  <div class="foot-r"><a href="https://github.com/nikhilsai0803/ai-vs-real-image-classifier" target="_blank">GitHub ↗</a></div>
</div>
""", unsafe_allow_html=True)
