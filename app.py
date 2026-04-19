"""
app.py — AI vs Real Image Classifier · Multi-Page Streamlit App
Pages: Detector · About · Tech Stack · How It Works
"""

import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import io, numpy as np, keras
from tensorflow.keras import applications
from PIL import Image
import streamlit as st

st.set_page_config(
    page_title="AI vs Real · Classifier",
    page_icon="🧿",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500;600&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#060609;--surface:#0d0d15;--card:#111120;--border:#1c1c2e;--border2:#28283c;
  --text:#e2e2f0;--muted:#52527a;--accent:#00f5d4;--purple:#8b5cf6;
  --red:#f43f5e;--green:#10b981;--amber:#f59e0b;--blue:#3b82f6;
  --head:'Syne',sans-serif;--mono:'JetBrains Mono',monospace;
}
#MainMenu,footer,header{visibility:hidden}
.stDeployButton{display:none!important}
.block-container{padding:1.5rem 0 0 0!important;max-width:100%!important}
.stApp{background:var(--bg)!important}
section[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important}
::-webkit-scrollbar{width:3px}
::-webkit-scrollbar-thumb{background:var(--border2)}

.nav-logo{font-family:var(--head);font-size:1.1rem;font-weight:800;color:var(--text);
  letter-spacing:-0.02em;padding:1rem 0 0.3rem;line-height:1.2}
.nav-logo span{color:var(--accent)}
.nav-subtitle{font-family:var(--mono);font-size:0.58rem;letter-spacing:0.18em;color:var(--muted);
  text-transform:uppercase;padding-bottom:1.4rem;border-bottom:1px solid var(--border);margin-bottom:1.2rem}
.nav-section{font-family:var(--mono);font-size:0.52rem;letter-spacing:0.22em;
  color:var(--muted);text-transform:uppercase;margin:1.2rem 0 0.6rem}

.stRadio>label{display:none!important}
.stRadio>div{display:flex!important;flex-direction:column!important;gap:0.2rem!important}
.stRadio>div>label{font-family:var(--mono)!important;font-size:0.78rem!important;
  color:var(--muted)!important;background:transparent!important;border:1px solid transparent!important;
  border-radius:3px!important;padding:0.5rem 0.8rem!important;cursor:pointer!important;
  transition:all 0.15s ease!important;letter-spacing:0.05em!important;display:flex!important;
  align-items:center!important;width:100%!important}
.stRadio>div>label:hover{color:var(--text)!important;background:var(--card)!important;border-color:var(--border2)!important}
.stRadio>div>label>div{display:none!important}
.stRadio>div>label>div+div{display:block!important;color:inherit!important}

.page-header{padding:2rem 3rem 1.6rem;border-bottom:1px solid var(--border);
  background:var(--surface);position:relative;overflow:hidden}
.page-header::after{content:'';position:absolute;top:-100px;right:-100px;width:350px;height:350px;
  background:radial-gradient(circle,rgba(0,245,212,0.05) 0%,transparent 65%);pointer-events:none}
.page-eyebrow{font-family:var(--mono);font-size:0.6rem;letter-spacing:0.25em;color:var(--accent);
  text-transform:uppercase;margin-bottom:0.5rem;display:flex;align-items:center;gap:0.5rem}
.page-eyebrow::before{content:'';display:inline-block;width:14px;height:1px;background:var(--accent)}
.page-h1{font-family:var(--head);font-size:clamp(1.8rem,3.5vw,2.8rem);font-weight:800;
  color:var(--text);letter-spacing:-0.025em;line-height:1.1;margin-bottom:0.5rem}
.page-h1 .hl{color:var(--accent)}
.page-desc{font-family:var(--mono);font-size:0.78rem;color:var(--muted);line-height:1.8;max-width:680px;letter-spacing:0.02em}
.content{padding:2rem 3rem}

.card{background:var(--card);border:1px solid var(--border);border-radius:4px;
  padding:1.5rem;margin-bottom:1rem;position:relative;overflow:hidden}
.card-accent{border-top:2px solid var(--accent)}
.card-purple{border-top:2px solid var(--purple)}
.card-green{border-top:2px solid var(--green)}
.card-amber{border-top:2px solid var(--amber)}
.card-blue{border-top:2px solid var(--blue)}
.card-red{border-top:2px solid var(--red)}
.card-title{font-family:var(--head);font-size:1rem;font-weight:700;color:var(--text);margin-bottom:0.5rem;letter-spacing:-0.01em}
.card-body{font-family:var(--mono);font-size:0.72rem;color:var(--muted);line-height:1.8;letter-spacing:0.02em}
.card-label{font-family:var(--mono);font-size:0.52rem;letter-spacing:0.2em;color:var(--muted);text-transform:uppercase;margin-bottom:0.6rem}

.stat-row{display:grid;grid-template-columns:repeat(4,1fr);gap:0.8rem;margin:1.4rem 0}
.stat-box{background:var(--card);border:1px solid var(--border);border-radius:4px;padding:1.2rem 1rem;text-align:center}
.stat-val{font-family:var(--head);font-size:1.8rem;font-weight:800;color:var(--text);letter-spacing:-0.03em;line-height:1;display:block;margin-bottom:0.3rem}
.stat-lbl{font-family:var(--mono);font-size:0.56rem;letter-spacing:0.18em;color:var(--muted);text-transform:uppercase}

.tag-row{display:flex;gap:0.4rem;flex-wrap:wrap;margin-top:0.8rem}
.tag{font-family:var(--mono);font-size:0.58rem;letter-spacing:0.1em;text-transform:uppercase;border-radius:2px;padding:0.2rem 0.55rem;border:1px solid}
.tag-accent{color:var(--accent);border-color:rgba(0,245,212,0.3);background:rgba(0,245,212,0.05)}
.tag-purple{color:var(--purple);border-color:rgba(139,92,246,0.3);background:rgba(139,92,246,0.05)}
.tag-green{color:var(--green);border-color:rgba(16,185,129,0.3);background:rgba(16,185,129,0.05)}
.tag-amber{color:var(--amber);border-color:rgba(245,158,11,0.3);background:rgba(245,158,11,0.05)}
.tag-blue{color:var(--blue);border-color:rgba(59,130,246,0.3);background:rgba(59,130,246,0.05)}

.divider{height:1px;background:var(--border);margin:1.8rem 0}
.section-title{font-family:var(--head);font-size:1.2rem;font-weight:700;color:var(--text);
  letter-spacing:-0.02em;margin-bottom:1rem;display:flex;align-items:center;gap:0.7rem}
.section-title::after{content:'';flex:1;height:1px;background:var(--border)}

.model-strip{border:1px solid var(--border);border-radius:3px;background:var(--card);
  padding:0.85rem 1rem;display:flex;align-items:center;gap:1rem;margin-bottom:1.2rem}
.live-dot{width:8px;height:8px;border-radius:50%;background:var(--accent);
  box-shadow:0 0 8px var(--accent);flex-shrink:0;animation:pulse 2s ease-in-out infinite}
@keyframes pulse{0%,100%{box-shadow:0 0 8px var(--accent)}50%{box-shadow:0 0 18px var(--accent),0 0 30px rgba(0,245,212,0.3)}}
.model-name-strip{font-family:var(--head);font-size:0.9rem;font-weight:700;color:var(--text);flex:1}

.result-card{border-radius:4px;background:var(--card);padding:1.6rem;position:relative;overflow:hidden;
  margin-bottom:1rem;animation:fadeUp 0.3s ease;border:1px solid var(--border2)}
@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.result-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px}
.result-card.real::before{background:var(--green)}
.result-card.fake::before{background:var(--red)}
.result-card.unsure::before{background:var(--amber)}

.verdict{font-family:var(--head);font-size:2.2rem;font-weight:800;letter-spacing:-0.025em;line-height:1;margin:0.3rem 0 0.2rem}
.verdict.real{color:var(--green)}.verdict.fake{color:var(--red)}.verdict.unsure{color:var(--amber)}
.conf-text{font-family:var(--mono);font-size:0.72rem;color:var(--muted);margin-bottom:1.2rem;letter-spacing:0.05em}

.bar-row{margin:0.3rem 0 0.8rem}
.bar-meta{display:flex;justify-content:space-between;font-family:var(--mono);font-size:0.62rem;color:var(--muted);margin-bottom:0.28rem}
.bar-track{height:4px;background:var(--border);border-radius:2px;overflow:hidden;margin-bottom:0.55rem}
.bar-fill{height:100%;border-radius:2px}
.bar-fill.g{background:var(--green)}.bar-fill.r{background:var(--red)}

.score-grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.55rem;margin-top:0.8rem}
.score-cell{background:var(--surface);border:1px solid var(--border);border-radius:3px;padding:0.6rem 0.8rem;text-align:center}
.score-val{font-family:var(--mono);font-size:0.95rem;font-weight:600;color:var(--text);display:block}
.score-lbl{font-family:var(--mono);font-size:0.52rem;letter-spacing:0.14em;color:var(--muted);text-transform:uppercase;display:block;margin-top:0.2rem}

.warn-badge{display:inline-flex;align-items:center;gap:0.4rem;font-family:var(--mono);font-size:0.62rem;
  letter-spacing:0.07em;color:var(--amber);border:1px solid rgba(245,158,11,0.3);
  background:rgba(245,158,11,0.05);border-radius:2px;padding:0.28rem 0.65rem;margin-bottom:0.8rem}

.img-frame{border:1px solid var(--border2);border-radius:4px;overflow:hidden;background:var(--card);margin-bottom:0.8rem}
.img-meta{font-family:var(--mono);font-size:0.6rem;letter-spacing:0.1em;color:var(--muted);
  text-transform:uppercase;padding:0.45rem 0.8rem;border-top:1px solid var(--border)}

.empty-state{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:320px;text-align:center;gap:0.8rem}
.empty-glyph{font-size:2.8rem;opacity:0.2}
.empty-text{font-family:var(--mono);font-size:0.68rem;letter-spacing:0.16em;color:var(--muted);text-transform:uppercase}
.empty-sub{font-family:var(--mono);font-size:0.62rem;color:var(--muted);max-width:240px;line-height:1.8;letter-spacing:0.04em}

.step{display:flex;gap:1.2rem;padding:1.2rem;background:var(--card);border:1px solid var(--border);border-radius:4px;margin-bottom:0.8rem;align-items:flex-start}
.step-num{font-family:var(--head);font-size:1.6rem;font-weight:800;color:var(--border2);line-height:1;min-width:2rem;text-align:center;padding-top:0.1rem}
.step-title{font-family:var(--head);font-size:0.95rem;font-weight:700;color:var(--text);margin-bottom:0.3rem}
.step-body{font-family:var(--mono);font-size:0.7rem;color:var(--muted);line-height:1.8}

.code-block{background:#08080f;border:1px solid var(--border);border-left:3px solid var(--accent);
  border-radius:3px;padding:1rem 1.2rem;font-family:var(--mono);font-size:0.7rem;color:#a0a0c8;
  line-height:1.9;overflow-x:auto;margin:0.8rem 0;white-space:pre}
.code-block .kw{color:var(--purple)}.code-block .str{color:var(--green)}
.code-block .cm{color:var(--muted)}.code-block .fn{color:var(--accent)}.code-block .num{color:var(--amber)}

.info-table{width:100%;border-collapse:collapse}
.info-table td{font-family:var(--mono);font-size:0.72rem;padding:0.6rem 0.8rem;border-bottom:1px solid var(--border);vertical-align:top}
.info-table td:first-child{color:var(--muted);width:38%;letter-spacing:0.06em;text-transform:uppercase;font-size:0.62rem}
.info-table td:last-child{color:var(--text)}

.footer{border-top:1px solid var(--border);padding:0.9rem 3rem;display:flex;justify-content:space-between;
  align-items:center;background:var(--surface);margin-top:2rem}
.footer-l{font-family:var(--mono);font-size:0.58rem;letter-spacing:0.14em;color:var(--muted);text-transform:uppercase}
.footer-r{font-family:var(--mono);font-size:0.58rem;color:var(--muted)}
.footer-r a{color:var(--accent);text-decoration:none}

.stSelectbox>div>div{background:var(--card)!important;border:1px solid var(--border2)!important;
  border-radius:3px!important;color:var(--text)!important;font-family:var(--mono)!important}
.stFileUploader>div{border:1px dashed var(--border2)!important;border-radius:4px!important;background:var(--card)!important}
.stFileUploader>div:hover{border-color:var(--accent)!important}
.stImage img{border-radius:0!important;display:block}
.stSpinner>div{border-top-color:var(--accent)!important}
div[data-testid="stSidebarUserContent"]{padding:0.5rem 1rem 2rem!important}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
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
    "MobileNetV2"    : {"params":"2.4M","speed":"Fastest","desc":"Lightweight depthwise separable convolutions. Best for real-time inference."},
    "EfficientNetB0" : {"params":"4.2M","speed":"Balanced","desc":"Compound scaling across depth, width & resolution. Best overall accuracy."},
    "NASNetMobile"   : {"params":"4.4M","speed":"Robust",  "desc":"Neural Architecture Search optimised. Most robust to edge cases."},
}

@st.cache_resource(show_spinner="Loading model weights…")
def load_model(model_name):
    path = MODEL_PATHS[model_name]
    if not os.path.exists(path): return None
    return keras.models.load_model(path)

def preprocess(pil_img, model_name):
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, 0)
    return MODEL_PREPROCESS[model_name](arr)

def run_prediction(model, tensor):
    score      = float(model.predict(tensor, verbose=0)[0][0])
    label      = "Real" if score >= 0.5 else "AI / Fake"
    confidence = score if score >= 0.5 else 1.0 - score
    return {
        "label"     : label,
        "confidence": round(confidence * 100, 2),
        "raw_score" : round(score, 4),
        "ai_pct"    : round((1.0 - score) * 100, 1),
        "real_pct"  : round(score * 100, 1),
        "uncertain" : confidence < UNCERTAIN_THRESH,
    }

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="nav-logo">AI<span>vs</span>Real<br>Classifier</div>
    <div class="nav-subtitle">Deep Learning · CV Project</div>
    <div class="nav-section">Navigation</div>
    """, unsafe_allow_html=True)

    page = st.selectbox(
        "Go to page",
        ["🧿  Detector", "📖  About Project", "⚙️  Tech Stack", "🔬  How It Works"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <div class="divider"></div>
    <div style="font-family:var(--mono);font-size:0.6rem;color:var(--muted);letter-spacing:0.12em;line-height:1.9;">
      <div style="color:var(--accent);margin-bottom:0.4rem;">STATUS</div>
      <div>TensorFlow 2.21</div><div>3 Models ready</div><div>Binary classifier</div>
    </div>
    <div class="divider"></div>
    <div style="font-family:var(--mono);font-size:0.58rem;color:var(--muted);letter-spacing:0.1em;">
      <a href="https://github.com/nikhilsai0803/ai-vs-real-image-classifier" target="_blank"
         style="color:var(--accent);text-decoration:none;">GitHub ↗</a>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 1 — DETECTOR
# ══════════════════════════════════════════════════════════════
if page == "🧿  Detector":
    st.markdown("""
    <div class="page-header">
      <div class="page-eyebrow">Live Inference Engine</div>
      <div class="page-h1">Image <span class="hl">Detector</span></div>
      <div class="page-desc">Upload any image and the model will classify it as AI-generated or a real photograph with a confidence score breakdown.</div>
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([1,1], gap="small")

    with col_l:
        st.markdown('<div style="padding:1.5rem 2rem;">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">01 · Select Model</div>', unsafe_allow_html=True)
        selected = st.selectbox("Model", list(MODEL_PATHS.keys()), label_visibility="collapsed")
        info = MODEL_INFO[selected]
        st.markdown(f"""
        <div class="model-strip">
          <div class="live-dot"></div>
          <div class="model-name-strip">{selected}</div>
          <span class="tag tag-accent">Fine-tuned</span>
          <span class="tag tag-purple">ImageNet</span>
        </div>
        <div style="font-family:var(--mono);font-size:0.68rem;color:var(--muted);line-height:1.7;letter-spacing:0.03em;margin-bottom:1.4rem;">
          {info['desc']} <span style="color:var(--text);margin-left:0.5rem;">{info['params']} params · {info['speed']}</span>
        </div>""", unsafe_allow_html=True)

        model = load_model(selected)
        if model is None:
            st.error(f"Model not found: `{MODEL_PATHS[selected]}`\nRun the training notebook first.")
            st.stop()

        st.markdown('<div class="card-label">02 · Upload Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("img", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")

        if uploaded:
            pil_img = Image.open(io.BytesIO(uploaded.read()))
            w, h = pil_img.size
            st.markdown('<div class="img-frame">', unsafe_allow_html=True)
            st.image(pil_img, use_container_width=True)
            st.markdown(f'<div class="img-meta">{uploaded.name} · {w}×{h}px · {pil_img.mode}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div style="padding:1.5rem 2rem;background:var(--surface);min-height:100%;">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">03 · Analysis Result</div>', unsafe_allow_html=True)

        if uploaded:
            with st.spinner("Running inference…"):
                tensor = preprocess(pil_img, selected)
                res    = run_prediction(model, tensor)
            label,conf = res["label"],res["confidence"]
            raw,uncertain = res["raw_score"],res["uncertain"]
            ai_pct,rpct = res["ai_pct"],res["real_pct"]
            if uncertain:   cls,icon,vc = "unsure","⚠️","unsure"
            elif label=="Real": cls,icon,vc = "real","✅","real"
            else:           cls,icon,vc = "fake","🤖","fake"
            warn = '<div class="warn-badge">⚠ LOW CONFIDENCE — result may be unreliable</div>' if uncertain else ""
            st.markdown(f"""
            <div class="result-card {cls}">
              <div style="font-family:var(--mono);font-size:0.55rem;letter-spacing:0.22em;color:var(--muted);text-transform:uppercase;margin-bottom:0.2rem;">Verdict</div>
              <div style="font-size:1.8rem;line-height:1;">{icon}</div>
              <div class="verdict {vc}">{label}</div>
              <div class="conf-text">Confidence: {conf}% &nbsp;·&nbsp; Raw: {raw:.4f}</div>
              {warn}
              <div class="bar-row">
                <div class="bar-meta"><span>🤖 AI-Generated</span><span>{ai_pct}%</span></div>
                <div class="bar-track"><div class="bar-fill r" style="width:{ai_pct}%"></div></div>
                <div class="bar-meta"><span>📷 Real Photo</span><span>{rpct}%</span></div>
                <div class="bar-track"><div class="bar-fill g" style="width:{rpct}%"></div></div>
              </div>
              <div class="score-grid3">
                <div class="score-cell"><span class="score-val">{raw:.4f}</span><span class="score-lbl">Raw Score</span></div>
                <div class="score-cell"><span class="score-val">{conf}%</span><span class="score-lbl">Confidence</span></div>
                <div class="score-cell"><span class="score-val" style="font-size:0.68rem;">{selected}</span><span class="score-lbl">Model</span></div>
              </div>
            </div>
            <div class="card" style="margin-top:0;">
              <div class="card-label">Interpretation</div>
              <div class="card-body">Score &gt; 0.50 → <span style="color:var(--green)">Real</span> &nbsp;|&nbsp; Score &lt; 0.50 → <span style="color:var(--red)">AI-Generated</span><br>Results below 85% confidence are flagged uncertain.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-glyph">🧿</div>
              <div class="empty-text">Awaiting input</div>
              <div class="empty-sub">Upload an image on the left panel to run the classifier</div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "📖  About Project":
    st.markdown("""
    <div class="page-header">
      <div class="page-eyebrow">Project Overview</div>
      <div class="page-h1">About <span class="hl">This Project</span></div>
      <div class="page-desc">A deep learning binary classifier that distinguishes AI-generated artwork from real photographs using transfer learning on three pretrained architectures.</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="content">', unsafe_allow_html=True)
    st.markdown("""
    <div class="stat-row">
      <div class="stat-box"><span class="stat-val" style="color:var(--accent);">~4.7K</span><span class="stat-lbl">Training Images</span></div>
      <div class="stat-box"><span class="stat-val" style="color:var(--purple);">3</span><span class="stat-lbl">Models Trained</span></div>
      <div class="stat-box"><span class="stat-val" style="color:var(--green);">2</span><span class="stat-lbl">Training Phases</span></div>
      <div class="stat-box"><span class="stat-val" style="color:var(--amber);">85%</span><span class="stat-lbl">Confidence Threshold</span></div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("""
        <div class="card card-accent">
          <div class="card-label">The Problem</div>
          <div class="card-title">Why Does This Matter?</div>
          <div class="card-body">AI-generated images have become indistinguishable from real photographs to the human eye. Tools like Midjourney, DALL·E, and Stable Diffusion produce art that raises concerns around misinformation, copyright, and digital trust.<br><br>This project builds a classifier that detects these differences at a feature level — patterns the human eye simply cannot perceive.</div>
          <div class="tag-row"><span class="tag tag-accent">Binary Classification</span><span class="tag tag-purple">Computer Vision</span></div>
        </div>
        <div class="card card-green">
          <div class="card-label">Dataset</div>
          <div class="card-title">Training Data</div>
          <div class="card-body">
            <table class="info-table">
              <tr><td>Source</td><td>Kaggle · tristanzhang32</td></tr>
              <tr><td>AI Images</td><td>~2,300 AI-generated artworks</td></tr>
              <tr><td>Real Images</td><td>~2,400 real photographs</td></tr>
              <tr><td>Formats</td><td>JPG · PNG · WEBP</td></tr>
              <tr><td>Split</td><td>70% Train · 15% Val · 15% Test</td></tr>
              <tr><td>Validation</td><td>Corrupt images removed at startup</td></tr>
            </table>
          </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card card-purple">
          <div class="card-label">Approach</div>
          <div class="card-title">Transfer Learning Strategy</div>
          <div class="card-body">Instead of training from scratch, we use models pre-trained on ImageNet — 1.2M images across 1,000 categories. These already know how to detect edges, textures, shapes, and complex visual patterns.<br><br>We then fine-tune them to learn the subtle artefacts separating AI art (synthetic gradients, GAN noise) from real photographs.</div>
          <div class="tag-row"><span class="tag tag-purple">Transfer Learning</span><span class="tag tag-green">Fine-tuning</span><span class="tag tag-amber">ImageNet</span></div>
        </div>
        <div class="card card-amber">
          <div class="card-label">Key Fixes Applied</div>
          <div class="card-title">What Made It Work</div>
          <div class="card-body">
            <table class="info-table">
              <tr><td>Preprocessing</td><td>Each model uses its own <code style="color:var(--accent)">preprocess_input</code> — no manual /255 scaling</td></tr>
              <tr><td>Fine-tuning</td><td>Only last 30 layers unfrozen — not the full base</td></tr>
              <tr><td>Callbacks</td><td>Fresh EarlyStopping per phase — stale state removed</td></tr>
              <tr><td>Validation</td><td>Corrupt images detected and removed before training</td></tr>
            </table>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Project Goals</div>', unsafe_allow_html=True)
    ca, cb, cc = st.columns(3, gap="medium")
    with ca:
        st.markdown("""<div class="card"><div style="font-size:1.5rem;margin-bottom:0.6rem;">🎯</div>
          <div class="card-title">Accurate Classification</div>
          <div class="card-body">Build a model that genuinely learns visual features — not just brightness or colour shortcuts — to reliably classify images.</div></div>""", unsafe_allow_html=True)
    with cb:
        st.markdown("""<div class="card"><div style="font-size:1.5rem;margin-bottom:0.6rem;">⚡</div>
          <div class="card-title">Lightweight & Fast</div>
          <div class="card-body">Use mobile-scale architectures so inference is fast even without a GPU. All three models run in seconds on CPU.</div></div>""", unsafe_allow_html=True)
    with cc:
        st.markdown("""<div class="card"><div style="font-size:1.5rem;margin-bottom:0.6rem;">🌐</div>
          <div class="card-title">Deployable UI</div>
          <div class="card-body">Ship a polished web interface anyone can use — no code required. Upload, click, get your answer instantly.</div></div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 3 — TECH STACK
# ══════════════════════════════════════════════════════════════
elif page == "⚙️  Tech Stack":
    st.markdown("""
    <div class="page-header">
      <div class="page-eyebrow">Tools & Libraries</div>
      <div class="page-h1">Tech <span class="hl">Stack</span></div>
      <div class="page-desc">Every library, framework, and tool used to build, train, evaluate, and deploy this project.</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="content">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Core ML Framework</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown("""<div class="card card-accent"><div style="font-size:1.6rem;margin-bottom:0.5rem;">🧠</div>
          <div class="card-title">TensorFlow 2.21</div>
          <div class="card-body">Open-source ML platform by Google. Provides the full model lifecycle: <code style="color:var(--accent)">tf.data</code> pipelines, training loops, model saving, and serving through Keras.</div>
          <div class="tag-row"><span class="tag tag-accent">Core</span><span class="tag tag-purple">Google</span></div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card card-purple"><div style="font-size:1.6rem;margin-bottom:0.5rem;">🔷</div>
          <div class="card-title">Keras</div>
          <div class="card-body">High-level API built into TensorFlow. Used to define architectures, compile, and run training. The <code style="color:var(--accent)">applications</code> module provides all three pretrained base models.</div>
          <div class="tag-row"><span class="tag tag-purple">API Layer</span><span class="tag tag-accent">Built-in</span></div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="card card-green"><div style="font-size:1.6rem;margin-bottom:0.5rem;">📦</div>
          <div class="card-title">NumPy</div>
          <div class="card-body">Fundamental array operations. Converts PIL images to float32 arrays, stacks batches, and post-processes raw model output scores before the UI displays them.</div>
          <div class="tag-row"><span class="tag tag-green">Numerical</span><span class="tag tag-amber">Fast</span></div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Pretrained Architectures</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown("""<div class="card card-blue"><div style="font-size:1.4rem;margin-bottom:0.5rem;">📱</div>
          <div class="card-title">MobileNetV2</div>
          <div class="card-body"><table class="info-table">
            <tr><td>Params</td><td>2.4M total</td></tr>
            <tr><td>Input range</td><td>[0,255] → [-1, 1]</td></tr>
            <tr><td>Design</td><td>Depthwise separable convolutions + inverted residuals</td></tr>
            <tr><td>Best for</td><td>Real-time, mobile inference</td></tr>
          </table></div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card card-accent"><div style="font-size:1.4rem;margin-bottom:0.5rem;">⚖️</div>
          <div class="card-title">EfficientNetB0</div>
          <div class="card-body"><table class="info-table">
            <tr><td>Params</td><td>4.2M total</td></tr>
            <tr><td>Input range</td><td>[0,255] → normalised internally</td></tr>
            <tr><td>Design</td><td>Compound scaling of depth, width & resolution</td></tr>
            <tr><td>Best for</td><td>Highest accuracy per parameter</td></tr>
          </table></div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="card card-purple"><div style="font-size:1.4rem;margin-bottom:0.5rem;">🔬</div>
          <div class="card-title">NASNetMobile</div>
          <div class="card-body"><table class="info-table">
            <tr><td>Params</td><td>4.4M total</td></tr>
            <tr><td>Input range</td><td>[0,255] → [-1, 1]</td></tr>
            <tr><td>Design</td><td>Neural Architecture Search — optimised by AI itself</td></tr>
            <tr><td>Best for</td><td>Robustness, generalisation</td></tr>
          </table></div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Supporting Libraries & Deployment</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("""<div class="card"><div class="card-label">Data & Visualisation</div>
          <table class="info-table">
            <tr><td>scikit-learn</td><td>train_test_split, classification_report, confusion_matrix</td></tr>
            <tr><td>Matplotlib</td><td>All training plots — curves, distributions, sample grids</td></tr>
            <tr><td>Seaborn</td><td>Styled confusion matrix heatmaps</td></tr>
            <tr><td>Pillow (PIL)</td><td>Image loading and format conversion in the app</td></tr>
          </table></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card"><div class="card-label">App & Deployment</div>
          <table class="info-table">
            <tr><td>Streamlit</td><td>Web app framework — all UI rendering and state management</td></tr>
            <tr><td>Git LFS</td><td>Storing large .keras model files in GitHub</td></tr>
            <tr><td>Streamlit Cloud</td><td>Free hosting — one-click deploy from GitHub</td></tr>
            <tr><td>packages.txt</td><td>System deps (libgl1) for Streamlit Cloud environment</td></tr>
          </table></div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">requirements.txt</div>', unsafe_allow_html=True)
    st.markdown("""<div class="code-block"><span class="cm"># pip install -r requirements.txt</span>

<span class="kw">tensorflow</span>==<span class="num">2.21.0</span>
<span class="kw">streamlit</span>&gt;=<span class="num">1.35.0</span>
<span class="kw">numpy</span>&gt;=<span class="num">1.24.0</span>
<span class="kw">Pillow</span>&gt;=<span class="num">10.0.0</span>
<span class="kw">scikit-learn</span>&gt;=<span class="num">1.3.0</span>
<span class="kw">matplotlib</span>&gt;=<span class="num">3.7.0</span>
<span class="kw">seaborn</span>&gt;=<span class="num">0.12.0</span></div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 4 — HOW IT WORKS
# ══════════════════════════════════════════════════════════════
elif page == "🔬  How It Works":
    st.markdown("""
    <div class="page-header">
      <div class="page-eyebrow">Technical Deep Dive</div>
      <div class="page-h1">How It <span class="hl">Works</span></div>
      <div class="page-desc">The full pipeline — from raw images on disk to a confident prediction — explained step by step.</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="content">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Training Pipeline</div>', unsafe_allow_html=True)

    steps = [
        ("Image Collection & Validation",
         "All images from <code style='color:var(--accent)'>AiArtData/</code> and <code style='color:var(--accent)'>RealArt/</code> are scanned recursively. Before training, every image is decoded by TensorFlow — corrupt files, truncated JPEGs, or images smaller than 10×10px are <strong style='color:var(--red)'>detected and removed</strong> so they never cause silent failures mid-training."),
        ("Data Split & tf.data Pipeline",
         "Images are shuffled and split 70/15/15. A <code style='color:var(--accent)'>tf.data.Dataset</code> pipeline loads images lazily. Training data gets light augmentation: random horizontal flip, ±10% brightness, ±15% contrast. The critical fix: images are cast to <code style='color:var(--accent)'>float32</code> in [0, 255] with <strong style='color:var(--red)'>no /255 division</strong> — each model's own preprocess_input handles that."),
        ("Phase 1 — Head Training",
         "The base model is <strong>fully frozen</strong>. Only the classification head trains: GlobalAveragePooling → BatchNorm → Dense(256, relu) → Dropout(0.4) → Dense(64, relu) → Dropout(0.2) → Sigmoid output. LR: 1e-3. EarlyStopping with patience=3. Typically converges in 5–9 epochs and reaches ~70–75% val accuracy."),
        ("Phase 2 — Surgical Fine-tuning",
         "Only the <strong>last 30 layers</strong> of the base model are unfrozen — not the whole network, which would destroy ImageNet weights. LR drops to 1e-5. A <strong>fresh EarlyStopping</strong> callback is created (the old one had stale state from Phase 1). This phase carefully adjusts high-level feature extractors to learn AI-art-specific patterns."),
        ("Evaluation & Saving",
         "All models are evaluated on the held-out test set. Confusion matrices, precision/recall/F1, and accuracy are reported. Every graph is saved to <code style='color:var(--accent)'>classifier_outputs/</code> — not the base directory. Models are saved as <code style='color:var(--accent)'>.keras</code> files for serving in this Streamlit app."),
    ]
    for i,(title,body) in enumerate(steps, 1):
        st.markdown(f"""<div class="step">
          <div class="step-num">0{i}</div>
          <div class="step-content">
            <div class="step-title">{title}</div>
            <div class="step-body">{body}</div>
          </div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Inference Pipeline (App)</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("""<div class="card card-accent">
          <div class="card-label">What Happens When You Upload</div>
          <div class="card-body" style="line-height:2.1;">
            1. PIL opens the image from memory (no disk write)<br>
            2. Resized to 224×224 px<br>
            3. Converted to float32 numpy array in [0, 255]<br>
            4. Model-specific <code style="color:var(--accent)">preprocess_input</code> scales values<br>
            5. Model returns a single sigmoid score in [0, 1]<br>
            6. Score ≥ 0.5 → Real &nbsp;|&nbsp; Score &lt; 0.5 → AI-Generated<br>
            7. Confidence = distance from decision boundary × 2<br>
            8. Result shown with bar breakdown
          </div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card card-red">
          <div class="card-label">The Critical Preprocessing Bug</div>
          <div class="card-body">
            The original code divided by 255 <em>before</em> calling <code style="color:var(--accent)">preprocess_input</code>.<br><br>
            Each model expects raw [0, 255] input:<br>
            <span style="color:var(--green);">MobileNetV2</span> → scales to [-1, 1]<br>
            <span style="color:var(--accent);">EfficientNetB0</span> → normalises internally<br>
            <span style="color:var(--purple);">NASNetMobile</span> → scales to [-1, 1]<br><br>
            Dividing by 255 first sent [0, 1] values through the wrong scaler — producing garbage activations.
            All models predicted only "Real" at 49% accuracy (pure guessing).
          </div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Key Code Fixes</div>', unsafe_allow_html=True)
    st.markdown("""<div class="code-block"><span class="cm"># ❌ WRONG — original (broke all 3 models, caused 49% accuracy)</span>
<span class="kw">def</span> <span class="fn">parse_image</span>(path, label):
    img = tf.image.<span class="fn">decode_image</span>(raw, channels=<span class="num">3</span>)
    img = tf.image.<span class="fn">resize</span>(img, [<span class="num">224</span>, <span class="num">224</span>])
    img = tf.<span class="fn">cast</span>(img, tf.float32) / <span class="num">255.0</span>  <span class="cm"># ← kills preprocess_input</span>
    <span class="kw">return</span> img, label

<span class="cm"># ✅ CORRECT — fixed version</span>
<span class="kw">def</span> <span class="fn">parse_image</span>(path, label):
    img = tf.image.<span class="fn">decode_image</span>(raw, channels=<span class="num">3</span>)
    img = tf.image.<span class="fn">resize</span>(img, [<span class="num">224</span>, <span class="num">224</span>])
    img = tf.<span class="fn">cast</span>(img, tf.float32)  <span class="cm"># cast only — NO /255</span>
    <span class="kw">return</span> img, label

<span class="cm"># ❌ WRONG fine-tuning — unfreezes ALL layers, destroys weights</span>
model.layers[<span class="num">1</span>].trainable = <span class="kw">True</span>

<span class="cm"># ✅ CORRECT — only last 30 layers unfrozen + fresh callbacks</span>
<span class="kw">for</span> layer <span class="kw">in</span> base.layers:       layer.trainable = <span class="kw">False</span>
<span class="kw">for</span> layer <span class="kw">in</span> base.layers[-<span class="num">30</span>:]: layer.trainable = <span class="kw">True</span>
cb_p2 = [keras.callbacks.<span class="fn">EarlyStopping</span>(monitor=<span class="str">'val_loss'</span>, patience=<span class="num">4</span>)]  <span class="cm"># fresh instance</span></div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <div class="footer-l">AI vs Real Image Classifier · TensorFlow · Transfer Learning · Streamlit</div>
  <div class="footer-r"><a href="https://github.com/YOUR_USERNAME/ai-vs-real-classifier" target="_blank">GitHub ↗</a></div>
</div>""", unsafe_allow_html=True)
