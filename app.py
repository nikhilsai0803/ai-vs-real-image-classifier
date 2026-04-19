"""
app.py — AI vs Real Image Classifier · Premium Multi-Page Streamlit App
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
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: #07070e;
  --surface: #0b0b16;
  --card: #0f0f1e;
  --card-hover: #13132a;
  --border: #1a1a2e;
  --border-hover: #2d2d52;
  --text: #eaeaf5;
  --muted: #4e4e72;
  --muted2: #6e6e98;
  --accent: #00f5d4;
  --accent-dim: rgba(0,245,212,0.08);
  --purple: #8b5cf6;
  --purple-dim: rgba(139,92,246,0.08);
  --red: #f43f5e;
  --red-dim: rgba(244,63,94,0.08);
  --green: #10b981;
  --green-dim: rgba(16,185,129,0.08);
  --amber: #f59e0b;
  --amber-dim: rgba(245,158,11,0.08);
  --blue: #3b82f6;
  --blue-dim: rgba(59,130,246,0.08);
  --head: 'Syne', sans-serif;
  --mono: 'JetBrains Mono', monospace;
  --ease: cubic-bezier(0.16, 1, 0.3, 1);
  --radius: 8px;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton, [data-testid="stToolbar"] { display: none !important; }
.stApp { background: var(--bg) !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stAppViewContainer"] > section { padding: 0 !important; }

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #06060f 0%, #080812 40%, #06060f 100%) !important;
  border-right: 1px solid rgba(0,245,212,0.08) !important;
  width: 240px !important;
  min-width: 240px !important;
  box-shadow: 4px 0 40px rgba(0,0,0,0.6) !important;
}
section[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }
div[data-testid="stSidebarUserContent"] { padding: 0 !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,245,212,0.2); border-radius: 2px; }

/* ── SIDEBAR LOGO BLOCK ── */
.sidebar-brand {
  padding: 1.8rem 1.4rem 1.4rem;
  position: relative;
  overflow: hidden;
}
.sidebar-brand::after {
  content: '';
  position: absolute;
  bottom: 0; left: 1.4rem; right: 1.4rem;
  height: 1px;
  background: linear-gradient(to right, rgba(0,245,212,0.3), transparent);
}
.sidebar-logo-eyebrow {
  font-family: var(--mono);
  font-size: 0.42rem;
  letter-spacing: 0.35em;
  color: var(--accent);
  text-transform: uppercase;
  margin-bottom: 0.6rem;
  opacity: 0.7;
}
.sidebar-logo {
  font-family: var(--head);
  font-size: 1.55rem;
  font-weight: 800;
  color: var(--text);
  letter-spacing: -0.04em;
  line-height: 1.05;
  margin-bottom: 0.5rem;
}
.sidebar-logo .ai { color: var(--text); }
.sidebar-logo .vs {
  color: var(--accent);
  font-style: italic;
  display: inline-block;
  text-shadow: 0 0 20px rgba(0,245,212,0.5);
}
.sidebar-logo .real { color: var(--text); }
.sidebar-logo-sub {
  font-family: var(--mono);
  font-size: 0.5rem;
  letter-spacing: 0.18em;
  color: var(--muted);
  text-transform: uppercase;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 0.4rem;
}
.sidebar-logo-sub .dot {
  width: 4px; height: 4px;
  border-radius: 50%;
  background: var(--accent);
  animation: logoPulse 2.5s ease-in-out infinite;
}
@keyframes logoPulse {
  0%, 100% { opacity: 0.4; transform: scale(1); }
  50% { opacity: 1; transform: scale(1.3); box-shadow: 0 0 6px rgba(0,245,212,0.8); }
}

/* ── SIDEBAR NAV ── */
.sidebar-nav-section {
  padding: 1.2rem 1rem;
}
.sidebar-nav-label {
  font-family: var(--mono);
  font-size: 0.42rem;
  letter-spacing: 0.3em;
  color: rgba(78,78,114,0.6);
  text-transform: uppercase;
  margin-bottom: 0.6rem;
  padding: 0 0.4rem;
}

/* Radio nav - completely restyled */
.stRadio > label { display: none !important; }
.stRadio > div { display: flex !important; flex-direction: column !important; gap: 0.25rem !important; }
.stRadio > div > label {
  font-family: var(--head) !important;
  font-size: 0.8rem !important;
  font-weight: 600 !important;
  color: rgba(110,110,152,0.8) !important;
  background: transparent !important;
  border: 1px solid transparent !important;
  border-radius: 6px !important;
  padding: 0.6rem 0.85rem !important;
  cursor: pointer !important;
  transition: all 0.2s cubic-bezier(0.16,1,0.3,1) !important;
  letter-spacing: -0.01em !important;
  position: relative !important;
}
.stRadio > div > label:hover {
  color: var(--text) !important;
  background: rgba(0,245,212,0.05) !important;
  border-color: rgba(0,245,212,0.15) !important;
  transform: translateX(3px) !important;
}
.stRadio > div > label[data-baseweb="radio"]:has(input:checked),
.stRadio > div > label:has(input:checked) {
  color: var(--accent) !important;
  background: rgba(0,245,212,0.07) !important;
  border-color: rgba(0,245,212,0.25) !important;
  box-shadow: inset 3px 0 0 var(--accent), 0 2px 12px rgba(0,245,212,0.08) !important;
}

/* ── SIDEBAR DIVIDER ── */
.sidebar-divider {
  height: 1px;
  margin: 0.2rem 1.4rem 0.2rem;
  background: linear-gradient(to right, rgba(0,245,212,0.12), transparent);
}

/* ── SIDEBAR STATUS ── */
.sidebar-status-wrap {
  padding: 1rem 1.4rem;
  position: relative;
}
.sidebar-status-label {
  font-family: var(--mono);
  font-size: 0.42rem;
  letter-spacing: 0.3em;
  color: var(--accent);
  text-transform: uppercase;
  margin-bottom: 0.8rem;
  opacity: 0.8;
}
.sidebar-status-item {
  font-family: var(--mono);
  font-size: 0.6rem;
  color: var(--muted2);
  line-height: 1;
  margin-bottom: 0.55rem;
  display: flex;
  align-items: center;
  gap: 0.55rem;
}
.status-dot {
  width: 5px; height: 5px;
  border-radius: 50%;
  background: var(--accent);
  flex-shrink: 0;
  animation: statusPulse 3s ease-in-out infinite;
}
.status-dot:nth-child(1) { animation-delay: 0s; }
.status-dot:nth-child(2) { animation-delay: 0.6s; }
.status-dot:nth-child(3) { animation-delay: 1.2s; }
@keyframes statusPulse {
  0%, 100% { opacity: 0.35; box-shadow: none; }
  50% { opacity: 1; box-shadow: 0 0 5px rgba(0,245,212,0.6); }
}

/* ── SIDEBAR BOTTOM ── */
.sidebar-bottom {
  position: absolute;
  bottom: 0; left: 0; right: 0;
  padding: 1.2rem 1.4rem;
  border-top: 1px solid rgba(0,245,212,0.06);
  background: linear-gradient(to top, rgba(0,245,212,0.03), transparent);
}
.sidebar-github {
  font-family: var(--mono);
  font-size: 0.58rem;
  color: var(--muted);
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.sidebar-github a {
  color: var(--accent);
  text-decoration: none;
  letter-spacing: 0.05em;
  transition: opacity 0.15s, letter-spacing 0.2s;
}
.sidebar-github a:hover { opacity: 0.7; letter-spacing: 0.1em; }
.sidebar-version-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  font-family: var(--mono);
  font-size: 0.45rem;
  letter-spacing: 0.12em;
  color: var(--muted);
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 0.2rem 0.45rem;
  margin-top: 0.6rem;
  text-transform: uppercase;
}

/* Page header */
.page-header {
  padding: 3rem 3.5rem 2.5rem;
  border-bottom: 1px solid var(--border);
  background: var(--surface);
  position: relative;
  overflow: hidden;
}
.page-header::before {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(ellipse 70% 80% at 85% 50%, rgba(0,245,212,0.04) 0%, transparent 60%);
  pointer-events: none;
}
.page-eyebrow {
  font-family: var(--mono);
  font-size: 0.55rem;
  letter-spacing: 0.3em;
  color: var(--accent);
  text-transform: uppercase;
  margin-bottom: 0.75rem;
  display: flex;
  align-items: center;
  gap: 0.6rem;
}
.page-eyebrow::before { content: ''; width: 20px; height: 1px; background: var(--accent); }
.page-h1 {
  font-family: var(--head);
  font-size: clamp(2.4rem, 5vw, 3.6rem);
  font-weight: 800;
  color: var(--text);
  letter-spacing: -0.035em;
  line-height: 1;
  margin-bottom: 0.8rem;
}
.page-h1 .hl { color: var(--accent); }
.page-desc {
  font-family: var(--mono);
  font-size: 0.72rem;
  color: var(--muted2);
  line-height: 2;
  max-width: 600px;
  letter-spacing: 0.02em;
}

.content { padding: 2.5rem 3.5rem 3rem; }

/* Cards with hover */
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.6rem;
  margin-bottom: 1rem;
  position: relative;
  overflow: hidden;
  transition: background 0.2s var(--ease), border-color 0.2s var(--ease), transform 0.2s var(--ease), box-shadow 0.2s var(--ease);
  cursor: default;
}
.card:hover {
  background: var(--card-hover);
  border-color: var(--border-hover);
  transform: translateY(-2px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.card-accent  { border-top: 2px solid var(--accent); }
.card-purple  { border-top: 2px solid var(--purple); }
.card-green   { border-top: 2px solid var(--green); }
.card-amber   { border-top: 2px solid var(--amber); }
.card-blue    { border-top: 2px solid var(--blue); }
.card-red     { border-top: 2px solid var(--red); }
.card-accent:hover  { box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(0,245,212,0.15); }
.card-purple:hover  { box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(139,92,246,0.15); }
.card-green:hover   { box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(16,185,129,0.15); }
.card-amber:hover   { box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(245,158,11,0.15); }
.card-blue:hover    { box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(59,130,246,0.15); }
.card-red:hover     { box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(244,63,94,0.15); }

.card-title { font-family: var(--head); font-size: 1.05rem; font-weight: 700; color: var(--text); margin-bottom: 0.55rem; letter-spacing: -0.01em; }
.card-body { font-family: var(--mono); font-size: 0.7rem; color: var(--muted2); line-height: 1.9; letter-spacing: 0.02em; }
.card-label { font-family: var(--mono); font-size: 0.48rem; letter-spacing: 0.25em; color: var(--muted); text-transform: uppercase; margin-bottom: 0.7rem; }

/* Stat boxes */
.stat-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2rem; }
.stat-box {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.5rem 1rem;
  text-align: center;
  transition: background 0.2s var(--ease), border-color 0.2s var(--ease), transform 0.2s var(--ease), box-shadow 0.2s var(--ease);
  cursor: default;
}
.stat-box:hover {
  background: var(--card-hover);
  border-color: var(--border-hover);
  transform: translateY(-3px);
  box-shadow: 0 12px 40px rgba(0,0,0,0.5);
}
.stat-val { font-family: var(--head); font-size: 2.2rem; font-weight: 800; letter-spacing: -0.04em; line-height: 1; display: block; margin-bottom: 0.4rem; }
.stat-lbl { font-family: var(--mono); font-size: 0.5rem; letter-spacing: 0.2em; color: var(--muted); text-transform: uppercase; }

/* Tags */
.tag-row { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-top: 1rem; }
.tag { font-family: var(--mono); font-size: 0.52rem; letter-spacing: 0.12em; text-transform: uppercase; border-radius: 4px; padding: 0.25rem 0.65rem; border: 1px solid; }
.tag-accent { color: var(--accent); border-color: rgba(0,245,212,0.25); background: var(--accent-dim); }
.tag-purple { color: var(--purple); border-color: rgba(139,92,246,0.25); background: var(--purple-dim); }
.tag-green  { color: var(--green);  border-color: rgba(16,185,129,0.25);  background: var(--green-dim); }
.tag-amber  { color: var(--amber);  border-color: rgba(245,158,11,0.25);  background: var(--amber-dim); }
.tag-blue   { color: var(--blue);   border-color: rgba(59,130,246,0.25);  background: var(--blue-dim); }

.divider { height: 1px; background: var(--border); margin: 2.5rem 0; }
.section-title {
  font-family: var(--head);
  font-size: 1.3rem;
  font-weight: 700;
  color: var(--text);
  letter-spacing: -0.02em;
  margin-bottom: 1.2rem;
  display: flex;
  align-items: center;
  gap: 0.9rem;
}
.section-title::after { content: ''; flex: 1; height: 1px; background: linear-gradient(to right, var(--border-hover), transparent); }

/* Model strip */
.model-strip {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--card);
  padding: 0.9rem 1.2rem;
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1.2rem;
  transition: border-color 0.2s, background 0.2s;
}
.model-strip:hover { border-color: rgba(0,245,212,0.2); background: var(--card-hover); }
.live-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--accent);
  flex-shrink: 0;
  animation: livePulse 2s ease-in-out infinite;
}
@keyframes livePulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(0,245,212,0.4); }
  50%       { box-shadow: 0 0 0 6px rgba(0,245,212,0); }
}
.model-name { font-family: var(--head); font-size: 0.9rem; font-weight: 700; color: var(--text); flex: 1; }

/* Result card */
.result-card {
  border-radius: var(--radius);
  background: var(--card);
  padding: 2rem;
  position: relative;
  overflow: hidden;
  border: 1px solid var(--border);
  animation: slideUp 0.35s var(--ease) forwards;
  margin-bottom: 1rem;
}
@keyframes slideUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}
.result-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; border-radius: var(--radius) var(--radius) 0 0; }
.result-card.real::before   { background: linear-gradient(to right, var(--green), transparent); }
.result-card.fake::before   { background: linear-gradient(to right, var(--red), transparent); }
.result-card.unsure::before { background: linear-gradient(to right, var(--amber), transparent); }
.result-card.real   { border-color: rgba(16,185,129,0.2); background: linear-gradient(135deg, rgba(16,185,129,0.04) 0%, var(--card) 50%); }
.result-card.fake   { border-color: rgba(244,63,94,0.2);  background: linear-gradient(135deg, rgba(244,63,94,0.04) 0%, var(--card) 50%); }
.result-card.unsure { border-color: rgba(245,158,11,0.2); background: linear-gradient(135deg, rgba(245,158,11,0.04) 0%, var(--card) 50%); }

.verdict-label { font-family: var(--mono); font-size: 0.5rem; letter-spacing: 0.28em; color: var(--muted); text-transform: uppercase; margin-bottom: 0.3rem; }
.verdict-icon { font-size: 2rem; line-height: 1; margin-bottom: 0.3rem; }
.verdict-text { font-family: var(--head); font-size: 2.8rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1; margin-bottom: 0.3rem; }
.verdict-text.real   { color: var(--green); }
.verdict-text.fake   { color: var(--red); }
.verdict-text.unsure { color: var(--amber); }
.verdict-sub { font-family: var(--mono); font-size: 0.65rem; color: var(--muted2); margin-bottom: 1.5rem; letter-spacing: 0.05em; }

.bar-section { margin-bottom: 1.2rem; }
.bar-row-header { display: flex; justify-content: space-between; font-family: var(--mono); font-size: 0.58rem; color: var(--muted2); margin-bottom: 0.35rem; }
.bar-track { height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; margin-bottom: 0.7rem; }
.bar-fill { height: 100%; border-radius: 3px; }
.bar-fill.ai   { background: linear-gradient(to right, var(--red), #ff6b8a); }
.bar-fill.real { background: linear-gradient(to right, var(--green), #34d399); }

.score-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.65rem; margin-top: 1rem; }
.score-cell {
  background: rgba(255,255,255,0.025);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.75rem;
  text-align: center;
  transition: border-color 0.15s, background 0.15s;
}
.score-cell:hover { border-color: var(--border-hover); background: rgba(255,255,255,0.04); }
.score-val { font-family: var(--mono); font-size: 1rem; font-weight: 600; color: var(--text); display: block; }
.score-lbl { font-family: var(--mono); font-size: 0.46rem; letter-spacing: 0.16em; color: var(--muted); text-transform: uppercase; display: block; margin-top: 0.3rem; }

.warn-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  font-family: var(--mono);
  font-size: 0.58rem;
  letter-spacing: 0.06em;
  color: var(--amber);
  border: 1px solid rgba(245,158,11,0.25);
  background: var(--amber-dim);
  border-radius: 5px;
  padding: 0.35rem 0.75rem;
  margin-bottom: 1rem;
}

.img-frame { border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; background: var(--card); margin-bottom: 1rem; transition: border-color 0.2s; }
.img-frame:hover { border-color: var(--border-hover); }
.img-meta { font-family: var(--mono); font-size: 0.56rem; letter-spacing: 0.12em; color: var(--muted); text-transform: uppercase; padding: 0.5rem 1rem; border-top: 1px solid var(--border); }

.empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 400px; text-align: center; gap: 1rem; }
.empty-ring { width: 80px; height: 80px; border-radius: 50%; border: 1px solid var(--border); display: flex; align-items: center; justify-content: center; font-size: 2rem; opacity: 0.25; }
.empty-title { font-family: var(--mono); font-size: 0.62rem; letter-spacing: 0.2em; color: var(--muted); text-transform: uppercase; }
.empty-sub { font-family: var(--mono); font-size: 0.58rem; color: var(--muted); max-width: 200px; line-height: 1.9; }

.det-left  { padding: 2rem 2.5rem; }
.det-right { padding: 2rem 2.5rem; background: var(--surface); min-height: calc(100vh - 200px); border-left: 1px solid var(--border); }

/* Steps */
.step {
  display: flex;
  gap: 1.4rem;
  padding: 1.4rem;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  margin-bottom: 0.8rem;
  align-items: flex-start;
  transition: background 0.2s, border-color 0.2s, transform 0.2s, box-shadow 0.2s;
}
.step:hover { background: var(--card-hover); border-color: var(--border-hover); transform: translateX(4px); box-shadow: -4px 0 0 0 var(--accent); }
.step-num { font-family: var(--head); font-size: 1.8rem; font-weight: 800; color: var(--border-hover); line-height: 1; min-width: 2.5rem; text-align: center; padding-top: 0.05rem; transition: color 0.2s; }
.step:hover .step-num { color: var(--accent); }
.step-title { font-family: var(--head); font-size: 0.95rem; font-weight: 700; color: var(--text); margin-bottom: 0.35rem; }
.step-body { font-family: var(--mono); font-size: 0.68rem; color: var(--muted2); line-height: 1.9; }

/* Code */
.code-block {
  background: #05050d;
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: var(--radius);
  padding: 1.2rem 1.5rem;
  font-family: var(--mono);
  font-size: 0.67rem;
  color: #9090c0;
  line-height: 2;
  overflow-x: auto;
  margin: 1rem 0;
  white-space: pre;
}
.code-block .kw  { color: var(--purple); }
.code-block .str { color: var(--green); }
.code-block .cm  { color: var(--muted); }
.code-block .fn  { color: var(--accent); }
.code-block .num { color: var(--amber); }

.info-table { width: 100%; border-collapse: collapse; }
.info-table td { font-family: var(--mono); font-size: 0.68rem; padding: 0.65rem 0.8rem; border-bottom: 1px solid var(--border); vertical-align: top; transition: background 0.15s; }
.info-table tr:hover td { background: rgba(255,255,255,0.02); }
.info-table td:first-child { color: var(--muted); width: 34%; letter-spacing: 0.08em; text-transform: uppercase; font-size: 0.58rem; }
.info-table td:last-child { color: var(--text); }

.footer { border-top: 1px solid var(--border); padding: 1.2rem 3.5rem; display: flex; justify-content: space-between; align-items: center; background: var(--surface); margin-top: 3rem; }
.footer-l { font-family: var(--mono); font-size: 0.54rem; letter-spacing: 0.15em; color: var(--muted); text-transform: uppercase; }
.footer-r { font-family: var(--mono); font-size: 0.54rem; color: var(--muted); }
.footer-r a { color: var(--accent); text-decoration: none; }
.footer-r a:hover { opacity: 0.7; }

/* Widget overrides */
.stSelectbox > div > div {
  background: var(--card) !important;
  border: 1px solid var(--border-hover) !important;
  border-radius: 6px !important;
  color: var(--text) !important;
  font-family: var(--mono) !important;
  font-size: 0.75rem !important;
}
.stSelectbox > div > div:hover { border-color: rgba(0,245,212,0.3) !important; }
.stFileUploader > div { border: 1px dashed var(--border-hover) !important; border-radius: var(--radius) !important; background: var(--card) !important; transition: border-color 0.2s, background 0.2s !important; }
.stFileUploader > div:hover { border-color: var(--accent) !important; background: var(--accent-dim) !important; }
.stImage img { border-radius: 0 !important; display: block; }
.stSpinner > div { border-top-color: var(--accent) !important; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

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
    "MobileNetV2"    : {"params": "2.4M", "speed": "Fastest",  "desc": "Depthwise separable convolutions. Optimised for real-time mobile inference."},
    "EfficientNetB0" : {"params": "4.2M", "speed": "Balanced", "desc": "Compound scaling of depth, width & resolution. Best accuracy per parameter."},
    "NASNetMobile"   : {"params": "4.4M", "speed": "Robust",   "desc": "Neural Architecture Search optimised. Most robust to edge cases and noise."},
}


@st.cache_resource(show_spinner="Loading model weights…")
def load_model(name):
    path = MODEL_PATHS[name]
    if not os.path.exists(path):
        return None
    return keras.models.load_model(path)


def preprocess(pil_img, name):
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, 0)
    return MODEL_PREPROCESS[name](arr)


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
    <div class="sidebar-brand">
      <div class="sidebar-logo-eyebrow">Computer Vision</div>
      <div class="sidebar-logo">
        <span class="ai">AI</span><span class="vs">vs</span><span class="real">Real</span><br>
        <span style="font-size:1.2rem;letter-spacing:-0.02em;color:rgba(234,234,245,0.5)">Classifier</span>
      </div>
      <div class="sidebar-logo-sub">
        <span class="dot"></span>
        <span>Deep Learning · CV Project</span>
      </div>
    </div>
    <div class="sidebar-nav-section">
      <div class="sidebar-nav-label">Navigation</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["🧿  Detector", "📖  About Project", "⚙️  Tech Stack", "🔬  How It Works"],
        label_visibility="collapsed",
        key="page_nav",
    )

    st.markdown("""
    <div style="padding: 0 1rem;">
      <div class="sidebar-divider"></div>
    </div>
    <div class="sidebar-status-wrap">
      <div class="sidebar-status-label">System Status</div>
      <div class="sidebar-status-item"><span class="status-dot"></span>TensorFlow 2.21</div>
      <div class="sidebar-status-item"><span class="status-dot"></span>3 Models ready</div>
      <div class="sidebar-status-item"><span class="status-dot"></span>Binary classifier</div>
    </div>
    <div style="padding: 0 1rem;">
      <div class="sidebar-divider"></div>
    </div>
    <div class="sidebar-status-wrap" style="padding-top:0.8rem;">
      <div class="sidebar-github"><a href="https://github.com/nikhilsai0803?tab=repositories" target="_blank">GitHub ↗</a></div>
      <div class="sidebar-version-badge">v1.0 · Streamlit</div>
    </div>
    """, unsafe_allow_html=True)


# PAGE 1 — DETECTOR
if page == "🧿  Detector":
    st.markdown("""
    <div class="page-header">
      <div class="page-eyebrow">Live Inference Engine</div>
      <div class="page-h1">Image <span class="hl">Detector</span></div>
      <div class="page-desc">Upload any image and the selected model will classify it as AI-generated or a real photograph, with a full confidence score breakdown.</div>
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="small")

    with col_l:
        st.markdown('<div class="det-left">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">01 · Select Model</div>', unsafe_allow_html=True)
        selected = st.selectbox("Model", list(MODEL_PATHS.keys()), label_visibility="collapsed")
        info = MODEL_INFO[selected]
        st.markdown(f"""
        <div class="model-strip">
          <div class="live-dot"></div>
          <div class="model-name">{selected}</div>
          <span class="tag tag-accent">Fine-tuned</span>
          <span class="tag tag-purple">ImageNet</span>
        </div>
        <p style="font-family:var(--mono);font-size:0.67rem;color:var(--muted2);line-height:1.8;margin-bottom:1.6rem;">
          {info['desc']}
          <span style="color:var(--text);margin-left:0.4rem;opacity:0.7;">{info['params']} params · {info['speed']}</span>
        </p>""", unsafe_allow_html=True)

        model = load_model(selected)
        if model is None:
            st.error(f"Model not found: `{MODEL_PATHS[selected]}`\nRun the training notebook first.")
            st.stop()

        st.markdown('<div class="card-label">02 · Upload Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("img", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")

        if uploaded:
            pil_img = Image.open(io.BytesIO(uploaded.read()))
            w, h = pil_img.size
            st.markdown('<div class="img-frame">', unsafe_allow_html=True)
            st.image(pil_img, use_container_width=True)
            st.markdown(f'<div class="img-meta">{uploaded.name} · {w}×{h}px · {pil_img.mode}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="det-right">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">03 · Analysis Result</div>', unsafe_allow_html=True)

        if uploaded:
            with st.spinner("Running inference…"):
                tensor = preprocess(pil_img, selected)
                res    = run_prediction(model, tensor)

            label, conf = res["label"], res["confidence"]
            raw, uncertain = res["raw_score"], res["uncertain"]
            ai_pct, rpct = res["ai_pct"], res["real_pct"]

            if uncertain:       cls, icon, vc = "unsure", "⚠️", "unsure"
            elif label == "Real": cls, icon, vc = "real", "✅", "real"
            else:               cls, icon, vc = "fake", "🤖", "fake"

            warn = '<div class="warn-badge">⚠&nbsp; LOW CONFIDENCE — result may be unreliable</div>' if uncertain else ""

            st.markdown(f"""
            <div class="result-card {cls}">
              <div class="verdict-label">Verdict</div>
              <div class="verdict-icon">{icon}</div>
              <div class="verdict-text {vc}">{label}</div>
              <div class="verdict-sub">Confidence: {conf}% &nbsp;·&nbsp; Raw: {raw:.4f} &nbsp;·&nbsp; {selected}</div>
              {warn}
              <div class="bar-section">
                <div class="bar-row-header"><span>🤖 AI-Generated</span><span>{ai_pct}%</span></div>
                <div class="bar-track"><div class="bar-fill ai" style="width:{ai_pct}%"></div></div>
                <div class="bar-row-header"><span>📷 Real Photo</span><span>{rpct}%</span></div>
                <div class="bar-track"><div class="bar-fill real" style="width:{rpct}%"></div></div>
              </div>
              <div class="score-grid">
                <div class="score-cell"><span class="score-val">{raw:.4f}</span><span class="score-lbl">Raw Score</span></div>
                <div class="score-cell"><span class="score-val">{conf}%</span><span class="score-lbl">Confidence</span></div>
                <div class="score-cell"><span class="score-val" style="font-size:0.65rem">{selected}</span><span class="score-lbl">Model</span></div>
              </div>
            </div>
            <div class="card">
              <div class="card-label">Interpretation Guide</div>
              <div class="card-body">
                Score &gt; 0.50 → <span style="color:var(--green)">Real Photo</span> &nbsp;|&nbsp;
                Score &lt; 0.50 → <span style="color:var(--red)">AI-Generated</span><br>
                Results below 85% confidence are flagged uncertain. Try multiple models for best accuracy.
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-ring">🧿</div>
              <div class="empty-title">Awaiting Input</div>
              <div class="empty-sub">Upload an image on the left to run the classifier</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# PAGE 2 — ABOUT
elif page == "📖  About Project":
    st.markdown("""
    <div class="page-header">
      <div class="page-eyebrow">Project Overview</div>
      <div class="page-h1">About <span class="hl">This Project</span></div>
      <div class="page-desc">A deep learning binary classifier that distinguishes AI-generated artwork from real photographs using transfer learning on three pretrained CNN architectures.</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="content">', unsafe_allow_html=True)
    st.markdown("""
    <div class="stat-row">
      <div class="stat-box"><span class="stat-val" style="color:var(--accent)">~4.7K</span><span class="stat-lbl">Training Images</span></div>
      <div class="stat-box"><span class="stat-val" style="color:var(--purple)">3</span><span class="stat-lbl">Models Trained</span></div>
      <div class="stat-box"><span class="stat-val" style="color:var(--green)">2</span><span class="stat-lbl">Training Phases</span></div>
      <div class="stat-box"><span class="stat-val" style="color:var(--amber)">85%</span><span class="stat-lbl">Confidence Threshold</span></div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("""
        <div class="card card-accent">
          <div class="card-label">The Problem</div>
          <div class="card-title">Why Does This Matter?</div>
          <div class="card-body">AI-generated images have become indistinguishable from real photographs to the human eye. Tools like Midjourney, DALL·E, and Stable Diffusion raise concerns around misinformation, copyright, and digital trust.<br><br>This classifier detects differences at a feature level — patterns the human eye simply cannot perceive.</div>
          <div class="tag-row"><span class="tag tag-accent">Binary Classification</span><span class="tag tag-purple">Computer Vision</span></div>
        </div>
        <div class="card card-green">
          <div class="card-label">Dataset</div>
          <div class="card-title">Training Data</div>
          <div class="card-body"><table class="info-table">
            <tr><td>Source</td><td>Kaggle · tristanzhang32</td></tr>
            <tr><td>AI Images</td><td>~2,300 AI-generated artworks</td></tr>
            <tr><td>Real Images</td><td>~2,400 real photographs</td></tr>
            <tr><td>Formats</td><td>JPG · PNG · WEBP</td></tr>
            <tr><td>Split</td><td>70% Train · 15% Val · 15% Test</td></tr>
            <tr><td>Validation</td><td>Corrupt images removed at startup</td></tr>
          </table></div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card card-purple">
          <div class="card-label">Approach</div>
          <div class="card-title">Transfer Learning Strategy</div>
          <div class="card-body">Instead of training from scratch, we use models pre-trained on ImageNet — 1.2M images across 1,000 categories. These already know edges, textures, shapes, and complex visual patterns.<br><br>We then fine-tune them to learn subtle artefacts separating AI art (synthetic gradients, GAN noise) from real photographs.</div>
          <div class="tag-row"><span class="tag tag-purple">Transfer Learning</span><span class="tag tag-green">Fine-tuning</span><span class="tag tag-amber">ImageNet</span></div>
        </div>
        <div class="card card-amber">
          <div class="card-label">Key Fixes</div>
          <div class="card-title">What Made It Work</div>
          <div class="card-body"><table class="info-table">
            <tr><td>Preprocessing</td><td>Each model uses its own <code style="color:var(--accent)">preprocess_input</code> — no manual /255</td></tr>
            <tr><td>Fine-tuning</td><td>Only last 30 layers unfrozen — not the full base</td></tr>
            <tr><td>Callbacks</td><td>Fresh EarlyStopping per phase — stale state removed</td></tr>
            <tr><td>Validation</td><td>Corrupt images detected and removed before training</td></tr>
          </table></div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div><div class="section-title">Project Goals</div>', unsafe_allow_html=True)
    ca, cb, cc = st.columns(3, gap="medium")
    with ca:
        st.markdown("""<div class="card card-accent"><div style="font-size:1.8rem;margin-bottom:0.8rem">🎯</div>
          <div class="card-title">Accurate Classification</div>
          <div class="card-body">Build a model that genuinely learns visual features — not just brightness or colour shortcuts — to reliably classify images.</div></div>""", unsafe_allow_html=True)
    with cb:
        st.markdown("""<div class="card card-purple"><div style="font-size:1.8rem;margin-bottom:0.8rem">⚡</div>
          <div class="card-title">Lightweight & Fast</div>
          <div class="card-body">Use mobile-scale architectures so inference is fast even without a GPU. All three models run in seconds on CPU.</div></div>""", unsafe_allow_html=True)
    with cc:
        st.markdown("""<div class="card card-green"><div style="font-size:1.8rem;margin-bottom:0.8rem">🌐</div>
          <div class="card-title">Deployable UI</div>
          <div class="card-body">Ship a polished web interface anyone can use — no code required. Upload, click, get your answer instantly.</div></div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# PAGE 3 — TECH STACK
elif page == "⚙️  Tech Stack":
    st.markdown("""
    <div class="page-header">
      <div class="page-eyebrow">Tools & Libraries</div>
      <div class="page-h1">Tech <span class="hl">Stack</span></div>
      <div class="page-desc">Every library, framework, and tool used to build, train, evaluate, and deploy this project.</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="content">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Core ML Framework</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown("""<div class="card card-accent"><div style="font-size:1.8rem;margin-bottom:0.7rem">🧠</div>
          <div class="card-title">TensorFlow 2.21</div>
          <div class="card-body">Open-source ML platform by Google. Provides the full model lifecycle: <code style="color:var(--accent)">tf.data</code> pipelines, training loops, model saving, and serving through Keras.</div>
          <div class="tag-row"><span class="tag tag-accent">Core</span><span class="tag tag-purple">Google</span></div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card card-purple"><div style="font-size:1.8rem;margin-bottom:0.7rem">🔷</div>
          <div class="card-title">Keras</div>
          <div class="card-body">High-level API built into TensorFlow. Used to define architectures, compile, and run training. The <code style="color:var(--accent)">applications</code> module provides all three pretrained base models.</div>
          <div class="tag-row"><span class="tag tag-purple">API Layer</span><span class="tag tag-accent">Built-in</span></div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="card card-green"><div style="font-size:1.8rem;margin-bottom:0.7rem">📦</div>
          <div class="card-title">NumPy</div>
          <div class="card-body">Fundamental array operations. Converts PIL images to float32 arrays, stacks batches, and post-processes raw model output scores before the UI displays them.</div>
          <div class="tag-row"><span class="tag tag-green">Numerical</span><span class="tag tag-amber">Fast</span></div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div><div class="section-title">Pretrained Architectures</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown("""<div class="card card-blue"><div style="font-size:1.6rem;margin-bottom:0.7rem">📱</div>
          <div class="card-title">MobileNetV2</div>
          <div class="card-body"><table class="info-table">
            <tr><td>Params</td><td>2.4M total</td></tr>
            <tr><td>Input</td><td>[0,255] → [-1, 1]</td></tr>
            <tr><td>Design</td><td>Depthwise separable + inverted residuals</td></tr>
            <tr><td>Best for</td><td>Real-time mobile inference</td></tr>
          </table></div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card card-accent"><div style="font-size:1.6rem;margin-bottom:0.7rem">⚖️</div>
          <div class="card-title">EfficientNetB0</div>
          <div class="card-body"><table class="info-table">
            <tr><td>Params</td><td>4.2M total</td></tr>
            <tr><td>Input</td><td>[0,255] → normalised internally</td></tr>
            <tr><td>Design</td><td>Compound scaling depth/width/res</td></tr>
            <tr><td>Best for</td><td>Highest accuracy per parameter</td></tr>
          </table></div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="card card-purple"><div style="font-size:1.6rem;margin-bottom:0.7rem">🔬</div>
          <div class="card-title">NASNetMobile</div>
          <div class="card-body"><table class="info-table">
            <tr><td>Params</td><td>4.4M total</td></tr>
            <tr><td>Input</td><td>[0,255] → [-1, 1]</td></tr>
            <tr><td>Design</td><td>Neural Architecture Search</td></tr>
            <tr><td>Best for</td><td>Robustness & generalisation</td></tr>
          </table></div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div><div class="section-title">Supporting Libraries & Deployment</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("""<div class="card"><div class="card-label">Data & Visualisation</div>
          <table class="info-table">
            <tr><td>scikit-learn</td><td>train_test_split, classification_report, confusion_matrix</td></tr>
            <tr><td>Matplotlib</td><td>Training plots — curves, distributions, sample grids</td></tr>
            <tr><td>Seaborn</td><td>Styled confusion matrix heatmaps</td></tr>
            <tr><td>Pillow (PIL)</td><td>Image loading and format conversion in the app</td></tr>
          </table></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card"><div class="card-label">App & Deployment</div>
          <table class="info-table">
            <tr><td>Streamlit</td><td>Web app framework — UI rendering and state management</td></tr>
            <tr><td>Git LFS</td><td>Storing large .keras model files in GitHub</td></tr>
            <tr><td>Streamlit Cloud</td><td>Free hosting — one-click deploy from GitHub</td></tr>
            <tr><td>packages.txt</td><td>System deps (libgl1) for Streamlit Cloud</td></tr>
          </table></div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div><div class="section-title">requirements.txt</div>', unsafe_allow_html=True)
    st.markdown("""<div class="code-block"><span class="cm"># pip install -r requirements.txt</span>

<span class="kw">tensorflow</span>==<span class="num">2.21.0</span>
<span class="kw">streamlit</span>&gt;=<span class="num">1.35.0</span>
<span class="kw">numpy</span>&gt;=<span class="num">1.24.0</span>
<span class="kw">Pillow</span>&gt;=<span class="num">10.0.0</span>
<span class="kw">scikit-learn</span>&gt;=<span class="num">1.3.0</span>
<span class="kw">matplotlib</span>&gt;=<span class="num">3.7.0</span>
<span class="kw">seaborn</span>&gt;=<span class="num">0.12.0</span></div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# PAGE 4 — HOW IT WORKS
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
         "All images from <code style='color:var(--accent)'>AiArtData/</code> and <code style='color:var(--accent)'>RealArt/</code> are scanned recursively. Every image is decoded by TensorFlow — corrupt files, truncated JPEGs, or images smaller than 10×10px are <strong style='color:var(--red)'>detected and removed</strong> so they never cause silent failures mid-training."),
        ("Data Split & tf.data Pipeline",
         "Images are shuffled and split 70/15/15. A <code style='color:var(--accent)'>tf.data.Dataset</code> pipeline loads images lazily. Training data gets light augmentation: random horizontal flip, ±10% brightness, ±15% contrast. Critical fix: images cast to <code style='color:var(--accent)'>float32</code> in [0, 255] with <strong style='color:var(--red)'>no /255 division</strong>."),
        ("Phase 1 — Head Training",
         "The base model is <strong>fully frozen</strong>. Only the classification head trains: GlobalAveragePooling → BatchNorm → Dense(256, relu) → Dropout(0.4) → Dense(64, relu) → Dropout(0.2) → Sigmoid output. LR: 1e-3, patience=3. Reaches ~70–75% val accuracy in 5–9 epochs."),
        ("Phase 2 — Surgical Fine-tuning",
         "Only the <strong>last 30 layers</strong> of the base model are unfrozen — not the whole network, which would destroy ImageNet weights. LR drops to 1e-5. A <strong>fresh EarlyStopping</strong> callback is created. This phase adjusts high-level feature extractors to learn AI-art-specific patterns."),
        ("Evaluation & Saving",
         "All models are evaluated on the held-out test set. Confusion matrices, precision/recall/F1, and accuracy are reported. Graphs saved to <code style='color:var(--accent)'>classifier_outputs/</code>. Models saved as <code style='color:var(--accent)'>.keras</code> files for serving in this app."),
    ]
    for i, (title, body) in enumerate(steps, 1):
        st.markdown(f"""<div class="step">
          <div class="step-num">0{i}</div>
          <div><div class="step-title">{title}</div><div class="step-body">{body}</div></div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div><div class="section-title">Inference Pipeline (App)</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("""<div class="card card-accent">
          <div class="card-label">What Happens When You Upload</div>
          <div class="card-body" style="line-height:2.2">
            1. PIL opens image from memory (no disk write)<br>
            2. Resized to 224×224 px<br>
            3. Converted to float32 numpy array in [0, 255]<br>
            4. Model-specific <code style="color:var(--accent)">preprocess_input</code> scales values<br>
            5. Model returns a single sigmoid score in [0, 1]<br>
            6. Score ≥ 0.5 → Real &nbsp;|&nbsp; &lt; 0.5 → AI-Generated<br>
            7. Confidence = distance from boundary × 2<br>
            8. Result shown with bar breakdown
          </div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card card-red">
          <div class="card-label">The Critical Preprocessing Bug</div>
          <div class="card-body">
            The original code divided by 255 <em>before</em> calling <code style="color:var(--accent)">preprocess_input</code>.<br><br>
            Each model expects raw [0, 255] input:<br>
            <span style="color:var(--green)">MobileNetV2</span> → scales to [-1, 1]<br>
            <span style="color:var(--accent)">EfficientNetB0</span> → normalises internally<br>
            <span style="color:var(--purple)">NASNetMobile</span> → scales to [-1, 1]<br><br>
            Dividing by 255 first produced garbage activations — all models predicted only "Real" at 49% accuracy.
          </div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div><div class="section-title">Key Code Fixes</div>', unsafe_allow_html=True)
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

<span class="cm"># ❌ WRONG — unfreezes ALL layers, destroys pretrained weights</span>
model.layers[<span class="num">1</span>].trainable = <span class="kw">True</span>

<span class="cm"># ✅ CORRECT — only last 30 layers + fresh callbacks per phase</span>
<span class="kw">for</span> layer <span class="kw">in</span> base.layers:        layer.trainable = <span class="kw">False</span>
<span class="kw">for</span> layer <span class="kw">in</span> base.layers[-<span class="num">30</span>:]:  layer.trainable = <span class="kw">True</span>
cb_p2 = [keras.callbacks.<span class="fn">EarlyStopping</span>(monitor=<span class="str">'val_loss'</span>, patience=<span class="num">4</span>)]</div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown("""
<div class="footer">
  <div class="footer-l">AI vs Real · TensorFlow · Transfer Learning · Streamlit</div>
  <div class="footer-r"><a href="https://github.com/nikhilsai0803?tab=repositories" target="_blank">GitHub ↗</a></div>
</div>""", unsafe_allow_html=True)