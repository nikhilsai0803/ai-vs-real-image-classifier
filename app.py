"""
app.py — AI vs Real Image Classifier · Multi-Page Streamlit App
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
    initial_sidebar_state="expanded",
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
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

:root {
  --bg:      #07070e;
  --surface: #0e0e1a;
  --card:    #13131f;
  --border:  #1f1f32;
  --border2: #2c2c45;
  --text:    #ededf5;
  --muted:   #555570;
  --dim:     #888899;
  --accent:  #00ffd1;
  --purple:  #9d6fff;
  --red:     #ff4d6d;
  --green:   #00e5a0;
  --amber:   #ffb830;
  --blue:    #4da6ff;
  --head:    'Syne', sans-serif;
  --mono:    'DM Mono', monospace;
}

/* ── Streamlit resets ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton, .stToolbar { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.stApp { background: var(--bg) !important; }
div[data-testid="stSidebarUserContent"] { padding: 1.5rem 1.2rem !important; }
section[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
  min-width: 220px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

/* ── Sidebar logo ── */
.logo {
  font-family: var(--head);
  font-size: 1.3rem;
  font-weight: 800;
  color: var(--text);
  letter-spacing: -0.03em;
  line-height: 1.1;
  margin-bottom: 0.3rem;
}
.logo em { color: var(--accent); font-style: normal; }
.logo-sub {
  font-family: var(--mono);
  font-size: 0.55rem;
  letter-spacing: 0.2em;
  color: var(--muted);
  text-transform: uppercase;
  margin-bottom: 1.5rem;
  padding-bottom: 1.5rem;
  border-bottom: 1px solid var(--border);
}

/* ── Nav buttons ── */
.nav-btn {
  display: block;
  width: 100%;
  text-align: left;
  font-family: var(--mono);
  font-size: 0.75rem;
  letter-spacing: 0.04em;
  color: var(--muted);
  background: transparent;
  border: 1px solid transparent;
  border-radius: 4px;
  padding: 0.55rem 0.8rem;
  margin-bottom: 0.3rem;
  cursor: pointer;
  transition: all 0.15s;
  text-decoration: none;
}
.nav-btn:hover { color: var(--text); background: var(--card); border-color: var(--border); }
.nav-btn.active { color: var(--accent); background: rgba(0,255,209,0.06); border-color: rgba(0,255,209,0.2); }

/* ── Status strip ── */
.status-strip {
  font-family: var(--mono);
  font-size: 0.6rem;
  letter-spacing: 0.1em;
  color: var(--muted);
  line-height: 2;
  margin-top: 1.5rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--border);
}
.status-head { color: var(--accent); font-size: 0.55rem; letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 0.4rem; }

/* ── Page header ── */
.ph {
  padding: 2.5rem 2.8rem 2rem;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  position: relative;
  overflow: hidden;
}
.ph::before {
  content: '';
  position: absolute; top: -80px; right: -80px;
  width: 300px; height: 300px;
  background: radial-gradient(circle, rgba(0,255,209,0.06) 0%, transparent 65%);
  pointer-events: none;
}
.ph-eye {
  font-family: var(--mono);
  font-size: 0.58rem;
  letter-spacing: 0.28em;
  color: var(--accent);
  text-transform: uppercase;
  margin-bottom: 0.6rem;
  display: flex; align-items: center; gap: 0.5rem;
}
.ph-eye::before { content: ''; display: inline-block; width: 20px; height: 1px; background: var(--accent); }
.ph-h1 {
  font-family: var(--head);
  font-size: clamp(2rem, 4vw, 3rem);
  font-weight: 800;
  color: var(--text);
  letter-spacing: -0.03em;
  line-height: 1.05;
  margin-bottom: 0.6rem;
}
.ph-h1 b { color: var(--accent); font-weight: 800; }
.ph-p {
  font-family: var(--mono);
  font-size: 0.76rem;
  color: var(--dim);
  line-height: 1.8;
  max-width: 600px;
}

/* ── Content wrapper ── */
.wrap { padding: 2rem 2.8rem; }

/* ── Cards ── */
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1.4rem 1.5rem;
  margin-bottom: 1rem;
}
.card.a { border-top: 2px solid var(--accent); }
.card.p { border-top: 2px solid var(--purple); }
.card.g { border-top: 2px solid var(--green); }
.card.r { border-top: 2px solid var(--red); }
.card.am{ border-top: 2px solid var(--amber); }
.card.b { border-top: 2px solid var(--blue); }
.card-lbl {
  font-family: var(--mono); font-size: 0.52rem; letter-spacing: 0.22em;
  color: var(--muted); text-transform: uppercase; margin-bottom: 0.55rem;
}
.card-h { font-family: var(--head); font-size: 1rem; font-weight: 700; color: var(--text); margin-bottom: 0.4rem; }
.card-p { font-family: var(--mono); font-size: 0.71rem; color: var(--dim); line-height: 1.85; }

/* ── Stats ── */
.stats { display: grid; grid-template-columns: repeat(4,1fr); gap: 0.8rem; margin: 1.5rem 0; }
.stat {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 6px; padding: 1.2rem; text-align: center;
}
.stat-v { font-family: var(--head); font-size: 2rem; font-weight: 800; line-height: 1; display: block; margin-bottom: 0.3rem; }
.stat-l { font-family: var(--mono); font-size: 0.54rem; letter-spacing: 0.16em; color: var(--muted); text-transform: uppercase; }

/* ── Tags ── */
.tags { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-top: 0.8rem; }
.tag {
  font-family: var(--mono); font-size: 0.56rem; letter-spacing: 0.1em;
  text-transform: uppercase; border-radius: 3px; padding: 0.2rem 0.55rem; border: 1px solid;
}
.ta { color: var(--accent); border-color: rgba(0,255,209,0.3); background: rgba(0,255,209,0.05); }
.tp { color: var(--purple); border-color: rgba(157,111,255,0.3); background: rgba(157,111,255,0.05); }
.tg { color: var(--green);  border-color: rgba(0,229,160,0.3); background: rgba(0,229,160,0.05); }
.tam{ color: var(--amber);  border-color: rgba(255,184,48,0.3); background: rgba(255,184,48,0.05); }
.tb { color: var(--blue);   border-color: rgba(77,166,255,0.3); background: rgba(77,166,255,0.05); }

/* ── Section heading ── */
.sec {
  font-family: var(--head); font-size: 1.15rem; font-weight: 700;
  color: var(--text); letter-spacing: -0.02em; margin-bottom: 1rem;
  display: flex; align-items: center; gap: 0.7rem;
}
.sec::after { content: ''; flex: 1; height: 1px; background: var(--border); }
.hr { height: 1px; background: var(--border); margin: 2rem 0; }

/* ── Detector specific ── */
.det-left  { padding: 1.8rem 2rem; border-right: 1px solid var(--border); }
.det-right { padding: 1.8rem 2rem; background: var(--surface); }

.model-bar {
  display: flex; align-items: center; gap: 0.9rem;
  border: 1px solid var(--border); border-radius: 5px;
  background: var(--card); padding: 0.8rem 1rem; margin-bottom: 1rem;
}
.dot {
  width: 9px; height: 9px; border-radius: 50%;
  background: var(--accent); box-shadow: 0 0 10px var(--accent);
  animation: glow 2s ease-in-out infinite; flex-shrink: 0;
}
@keyframes glow {
  0%,100% { box-shadow: 0 0 8px var(--accent); }
  50%      { box-shadow: 0 0 20px var(--accent), 0 0 35px rgba(0,255,209,0.25); }
}
.model-nm { font-family: var(--head); font-size: 0.88rem; font-weight: 700; color: var(--text); flex: 1; }
.badge {
  font-family: var(--mono); font-size: 0.52rem; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--accent);
  border: 1px solid rgba(0,255,209,0.25); border-radius: 3px;
  padding: 0.18rem 0.5rem; background: rgba(0,255,209,0.05);
}

.img-box { border: 1px solid var(--border2); border-radius: 5px; overflow: hidden; background: var(--card); margin-bottom: 0.8rem; }
.img-meta { font-family: var(--mono); font-size: 0.58rem; letter-spacing: 0.1em; color: var(--muted);
  text-transform: uppercase; padding: 0.45rem 0.8rem; border-top: 1px solid var(--border); }

.result {
  border: 1px solid var(--border2); border-radius: 6px;
  background: var(--card); padding: 1.5rem;
  position: relative; overflow: hidden;
  margin-bottom: 0.8rem;
  animation: up 0.3s ease;
}
@keyframes up { from { opacity:0; transform: translateY(8px); } to { opacity:1; transform: translateY(0); } }
.result::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; }
.result.real::before  { background: var(--green); }
.result.fake::before  { background: var(--red); }
.result.unsure::before{ background: var(--amber); }

.verdict-lbl { font-family: var(--mono); font-size: 0.52rem; letter-spacing: 0.22em; color: var(--muted); text-transform: uppercase; }
.verdict {
  font-family: var(--head); font-size: 2.4rem; font-weight: 800;
  letter-spacing: -0.03em; line-height: 1.05; margin: 0.2rem 0;
}
.verdict.real  { color: var(--green); }
.verdict.fake  { color: var(--red); }
.verdict.unsure{ color: var(--amber); }
.verdict-conf { font-family: var(--mono); font-size: 0.7rem; color: var(--muted); margin-bottom: 1.2rem; }

.brow { display:flex; justify-content:space-between; font-family:var(--mono); font-size:0.62rem; color:var(--muted); margin-bottom:0.3rem; }
.btrack { height:5px; background:var(--border); border-radius:3px; overflow:hidden; margin-bottom:0.6rem; }
.bfill  { height:100%; border-radius:3px; transition: width 0.5s ease; }
.bg { background: var(--green); }
.br { background: var(--red); }

.scores { display:grid; grid-template-columns:1fr 1fr 1fr; gap:0.55rem; margin-top:0.9rem; }
.sc { background:var(--surface); border:1px solid var(--border); border-radius:4px; padding:0.65rem; text-align:center; }
.sc-v { font-family:var(--mono); font-size:0.9rem; font-weight:500; color:var(--text); display:block; }
.sc-l { font-family:var(--mono); font-size:0.5rem; letter-spacing:0.14em; color:var(--muted); text-transform:uppercase; display:block; margin-top:0.2rem; }

.warn { display:inline-flex; align-items:center; gap:0.4rem; font-family:var(--mono); font-size:0.62rem;
  color:var(--amber); border:1px solid rgba(255,184,48,0.3); background:rgba(255,184,48,0.05);
  border-radius:3px; padding:0.3rem 0.7rem; margin-bottom:0.8rem; }

.empty { display:flex; flex-direction:column; align-items:center; justify-content:center;
  min-height:300px; text-align:center; gap:0.8rem; }
.empty-icon { font-size:3rem; opacity:0.15; }
.empty-t { font-family:var(--mono); font-size:0.68rem; letter-spacing:0.18em; color:var(--muted); text-transform:uppercase; }
.empty-s { font-family:var(--mono); font-size:0.62rem; color:var(--muted); max-width:220px; line-height:1.8; }

/* ── Steps ── */
.step { display:flex; gap:1.2rem; padding:1.2rem 1.3rem; background:var(--card);
  border:1px solid var(--border); border-radius:6px; margin-bottom:0.8rem; align-items:flex-start; }
.step-n { font-family:var(--head); font-size:1.8rem; font-weight:800; color:var(--border2);
  line-height:1; min-width:2.2rem; text-align:center; padding-top:0.05rem; }
.step-h { font-family:var(--head); font-size:0.95rem; font-weight:700; color:var(--text); margin-bottom:0.3rem; }
.step-b { font-family:var(--mono); font-size:0.7rem; color:var(--dim); line-height:1.85; }

/* ── Code ── */
.code {
  background: #060610; border:1px solid var(--border);
  border-left:3px solid var(--accent); border-radius:4px;
  padding:1.1rem 1.3rem; font-family:var(--mono); font-size:0.68rem;
  color:#9090b8; line-height:1.95; overflow-x:auto; white-space:pre; margin:0.8rem 0;
}
.kw  { color: var(--purple); }
.fn  { color: var(--accent); }
.st  { color: var(--green); }
.cm  { color: var(--muted); }
.nm  { color: var(--amber); }

/* ── Table ── */
.tbl { width:100%; border-collapse:collapse; }
.tbl td { font-family:var(--mono); font-size:0.7rem; padding:0.6rem 0.7rem; border-bottom:1px solid var(--border); vertical-align:top; }
.tbl td:first-child { color:var(--muted); width:36%; font-size:0.6rem; letter-spacing:0.07em; text-transform:uppercase; }
.tbl td:last-child { color:var(--text); }

/* ── Footer ── */
.foot {
  border-top:1px solid var(--border); padding:1rem 2.8rem;
  display:flex; justify-content:space-between; align-items:center;
  background:var(--surface); margin-top:2rem;
}
.foot-l { font-family:var(--mono); font-size:0.56rem; letter-spacing:0.14em; color:var(--muted); text-transform:uppercase; }
.foot-r { font-family:var(--mono); font-size:0.56rem; color:var(--muted); }
.foot-r a { color:var(--accent); text-decoration:none; }

/* ── Streamlit widget overrides ── */
.stSelectbox > div > div {
  background: var(--card) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 4px !important;
  color: var(--text) !important;
  font-family: var(--mono) !important;
  font-size: 0.82rem !important;
}
.stSelectbox label { color: var(--muted) !important; font-family: var(--mono) !important; font-size: 0.62rem !important; letter-spacing: 0.14em !important; text-transform: uppercase !important; }
.stFileUploader > div { border: 1px dashed var(--border2) !important; border-radius: 5px !important; background: var(--card) !important; }
.stFileUploader > div:hover { border-color: var(--accent) !important; }
.stFileUploader label { color: var(--muted) !important; font-family: var(--mono) !important; font-size: 0.62rem !important; letter-spacing: 0.14em !important; text-transform: uppercase !important; }
.stImage img { border-radius: 0 !important; display: block !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="logo">AI<em>vs</em>Real</div>
    <div class="logo-sub">Image Classifier · Deep Learning</div>
    """, unsafe_allow_html=True)

    pages = ["🧿 Detector", "📖 About", "⚙️ Tech Stack", "🔬 How It Works"]
    for p in pages:
        active = "active" if st.session_state.page in p or p in st.session_state.page else ""
        if st.button(p, key=f"nav_{p}", use_container_width=True):
            st.session_state.page = p
            st.rerun()

    st.markdown("""
    <div class="status-strip">
      <div class="status-head">System Status</div>
      <div>● TensorFlow 2.21</div>
      <div>● 3 Models available</div>
      <div>● Binary classifier</div>
      <div style="margin-top:1rem;">
        <a href="https://github.com/nikhilsai0803/ai-vs-real-image-classifier"
           target="_blank" style="color:var(--accent);text-decoration:none;font-size:0.62rem;">
          ↗ GitHub Repository
        </a>
      </div>
    </div>
    """, unsafe_allow_html=True)

page = st.session_state.page

# ══════════════════════════════════════════════════════════════
# PAGE 1 — DETECTOR
# ══════════════════════════════════════════════════════════════
if "Detector" in page:
    st.markdown("""
    <div class="ph">
      <div class="ph-eye">Live Inference Engine</div>
      <div class="ph-h1">Image <b>Detector</b></div>
      <div class="ph-p">Upload any image. The selected model classifies it as AI-generated or a real photograph with full confidence breakdown.</div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="small")

    with col_l:
        st.markdown('<div class="det-left">', unsafe_allow_html=True)

        st.markdown('<div class="card-lbl" style="margin-bottom:0.5rem;">01 · Select Model</div>', unsafe_allow_html=True)
        selected = st.selectbox("model", list(MODEL_PATHS.keys()), label_visibility="collapsed")

        info = MODEL_INFO[selected]
        st.markdown(f"""
        <div class="model-bar">
          <div class="dot"></div>
          <div class="model-nm">{selected}</div>
          <span class="badge">Fine-tuned</span>
          <span class="badge">ImageNet</span>
        </div>
        <div style="font-family:var(--mono);font-size:0.68rem;color:var(--muted);line-height:1.7;margin-bottom:1.4rem;">
          {info['desc']} &nbsp;<span style="color:var(--text)">{info['params']} params · {info['speed']}</span>
        </div>
        """, unsafe_allow_html=True)

        model = load_model(selected)
        if model is None:
            st.error(f"**Model not found:** `{MODEL_PATHS[selected]}`\n\nRun the training notebook to generate `classifier_outputs/`.")
            st.stop()

        st.markdown('<div class="card-lbl" style="margin-bottom:0.5rem;">02 · Upload Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")

        if uploaded:
            pil_img = Image.open(io.BytesIO(uploaded.read()))
            w, h = pil_img.size
            st.markdown('<div class="img-box">', unsafe_allow_html=True)
            st.image(pil_img, use_container_width=True)
            st.markdown(f'<div class="img-meta">{uploaded.name} &nbsp;·&nbsp; {w}×{h}px &nbsp;·&nbsp; {pil_img.mode}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="det-right">', unsafe_allow_html=True)
        st.markdown('<div class="card-lbl" style="margin-bottom:0.8rem;">03 · Result</div>', unsafe_allow_html=True)

        if uploaded:
            with st.spinner("Analysing…"):
                tensor = preprocess(pil_img, selected)
                res    = predict(model, tensor)

            label, conf    = res["label"], res["conf"]
            raw, uncertain = res["raw"], res["uncertain"]
            ai_pct, rpct   = res["ai_pct"], res["real_pct"]

            if uncertain:   cls, icon, vc = "unsure", "⚠️", "unsure"
            elif "Real" in label: cls, icon, vc = "real", "✅", "real"
            else:           cls, icon, vc = "fake", "🤖", "fake"

            warn = '<div class="warn">⚠ LOW CONFIDENCE — result may be unreliable</div>' if uncertain else ""

            st.markdown(f"""
            <div class="result {cls}">
              <div class="verdict-lbl">Verdict</div>
              <div style="font-size:2rem;margin:0.2rem 0;">{icon}</div>
              <div class="verdict {vc}">{label}</div>
              <div class="verdict-conf">Confidence: {conf}% &nbsp;·&nbsp; Raw score: {raw:.4f}</div>
              {warn}
              <div class="brow"><span>🤖 AI-Generated</span><span>{ai_pct}%</span></div>
              <div class="btrack"><div class="bfill br" style="width:{ai_pct}%"></div></div>
              <div class="brow"><span>📷 Real Photo</span><span>{rpct}%</span></div>
              <div class="btrack"><div class="bfill bg" style="width:{rpct}%"></div></div>
              <div class="scores">
                <div class="sc"><span class="sc-v">{raw:.4f}</span><span class="sc-l">Raw Score</span></div>
                <div class="sc"><span class="sc-v">{conf}%</span><span class="sc-l">Confidence</span></div>
                <div class="sc"><span class="sc-v" style="font-size:0.65rem;">{selected}</span><span class="sc-l">Model</span></div>
              </div>
            </div>
            <div class="card" style="margin-top:0;">
              <div class="card-lbl">How to read this</div>
              <div class="card-p">Score &gt; 0.5 → <span style="color:var(--green)">Real</span> &nbsp;|&nbsp; Score &lt; 0.5 → <span style="color:var(--red)">AI-Generated</span><br>
              Below 85% confidence is flagged as uncertain.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty">
              <div class="empty-icon">🧿</div>
              <div class="empty-t">Awaiting Image</div>
              <div class="empty-s">Upload a JPG, PNG, or WEBP on the left to run the classifier</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — ABOUT
# ══════════════════════════════════════════════════════════════
elif "About" in page:
    st.markdown("""
    <div class="ph">
      <div class="ph-eye">Project Overview</div>
      <div class="ph-h1">About <b>This Project</b></div>
      <div class="ph-p">A deep learning binary classifier distinguishing AI-generated artwork from real photographs using transfer learning on three pretrained architectures.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="wrap">', unsafe_allow_html=True)

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
        st.markdown("""<div class="card"><div style="font-size:1.6rem;margin-bottom:0.5rem;">🎯</div>
        <div class="card-h">Accurate Classification</div>
        <div class="card-p">Learn genuine visual features — not brightness or colour shortcuts — to reliably separate AI from real.</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card"><div style="font-size:1.6rem;margin-bottom:0.5rem;">⚡</div>
        <div class="card-h">Lightweight & Fast</div>
        <div class="card-p">Mobile-scale architectures so inference runs in seconds even on CPU — no GPU required.</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="card"><div style="font-size:1.6rem;margin-bottom:0.5rem;">🌐</div>
        <div class="card-h">Deployable UI</div>
        <div class="card-p">A polished web interface anyone can use — no code required. Upload, click, get your answer.</div></div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 3 — TECH STACK
# ══════════════════════════════════════════════════════════════
elif "Tech" in page:
    st.markdown("""
    <div class="ph">
      <div class="ph-eye">Tools & Libraries</div>
      <div class="ph-h1">Tech <b>Stack</b></div>
      <div class="ph-p">Every library, framework, and tool used to build, train, evaluate, and deploy this project.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="wrap">', unsafe_allow_html=True)
    st.markdown('<div class="sec">Core ML Framework</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown("""<div class="card a"><div style="font-size:1.5rem;margin-bottom:0.4rem;">🧠</div>
        <div class="card-h">TensorFlow 2.21</div>
        <div class="card-p">Full ML lifecycle — <code style="color:var(--accent)">tf.data</code> pipelines, training loops, model saving and serving. Open-source by Google.</div>
        <div class="tags"><span class="tag ta">Core</span><span class="tag tp">Google</span></div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card p"><div style="font-size:1.5rem;margin-bottom:0.4rem;">🔷</div>
        <div class="card-h">Keras</div>
        <div class="card-p">High-level API inside TensorFlow. Defines model architecture, compiles, runs training. The <code style="color:var(--accent)">applications</code> module provides all three pretrained bases.</div>
        <div class="tags"><span class="tag tp">API</span><span class="tag ta">Built-in</span></div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="card g"><div style="font-size:1.5rem;margin-bottom:0.4rem;">📦</div>
        <div class="card-h">NumPy</div>
        <div class="card-p">Converts PIL images to float32 arrays, stacks batches, post-processes raw sigmoid scores before display in the UI.</div>
        <div class="tags"><span class="tag tg">Numerical</span></div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec">Pretrained Architectures</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown("""<div class="card b"><div style="font-size:1.3rem;margin-bottom:0.4rem;">📱</div>
        <div class="card-h">MobileNetV2</div>
        <div class="card-p"><table class="tbl">
          <tr><td>Params</td><td>2.4M total</td></tr>
          <tr><td>Input</td><td>[0,255] → [-1, 1]</td></tr>
          <tr><td>Design</td><td>Depthwise separable + inverted residuals</td></tr>
          <tr><td>Best for</td><td>Real-time, mobile inference</td></tr>
        </table></div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card a"><div style="font-size:1.3rem;margin-bottom:0.4rem;">⚖️</div>
        <div class="card-h">EfficientNetB0</div>
        <div class="card-p"><table class="tbl">
          <tr><td>Params</td><td>4.2M total</td></tr>
          <tr><td>Input</td><td>[0,255] → normalised internally</td></tr>
          <tr><td>Design</td><td>Compound scaling of depth, width, resolution</td></tr>
          <tr><td>Best for</td><td>Highest accuracy per parameter</td></tr>
        </table></div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="card p"><div style="font-size:1.3rem;margin-bottom:0.4rem;">🔬</div>
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

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 4 — HOW IT WORKS
# ══════════════════════════════════════════════════════════════
elif "How" in page:
    st.markdown("""
    <div class="ph">
      <div class="ph-eye">Technical Deep Dive</div>
      <div class="ph-h1">How It <b>Works</b></div>
      <div class="ph-p">The full pipeline — from raw images on disk to a confident prediction — explained step by step.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="wrap">', unsafe_allow_html=True)
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
          <div>
            <div class="step-h">{title}</div>
            <div class="step-b">{body}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec">Inference Pipeline</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("""<div class="card a">
        <div class="card-lbl">What Happens When You Upload</div>
        <div class="card-p" style="line-height:2.2;">
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
          The /255 sent wrong values through the scaler — producing garbage activations. All models predicted only "Real" at <strong style="color:var(--red)">49% accuracy</strong> (random guessing).
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

    st.markdown('</div>', unsafe_allow_html=True)

# ─── Footer ────────────────────────────────────────────────────
st.markdown("""
<div class="foot">
  <div class="foot-l">AI vs Real Image Classifier &nbsp;·&nbsp; TensorFlow &nbsp;·&nbsp; Transfer Learning &nbsp;·&nbsp; Streamlit</div>
  <div class="foot-r"><a href="https://github.com/nikhilsai0803/ai-vs-real-image-classifier" target="_blank">GitHub ↗</a></div>
</div>
""", unsafe_allow_html=True)
