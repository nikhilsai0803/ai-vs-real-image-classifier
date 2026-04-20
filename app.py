"""
app.py — AI vs Real Image Classifier · Flask Edition
Converts Streamlit app to production-ready Flask with 4 models.
"""

import os
import io
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, redirect, url_for
import keras
from tensorflow.keras import applications

# ── App Init ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE         = 224
UNCERTAIN_THRESH = 0.85
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

# ── Model Registry ────────────────────────────────────────────────────────────
# Place .keras files in classifier_outputs/ next to app.py
MODEL_CONFIG = {
    "MobileNetV2": {
        "path":        "classifier_outputs/mobilenetv2.keras",
        "preprocess":  applications.mobilenet_v2.preprocess_input,
        "params":      "2.4M",
        "speed":       "Fastest",
        "color":       "#00f5d4",
        "desc":        "Depthwise separable convolutions optimised for real-time mobile inference.",
        "input_range": "[-1, 1]",
    },
    "EfficientNetB0": {
        "path":        "classifier_outputs/efficientnetb0.keras",
        "preprocess":  applications.efficientnet.preprocess_input,
        "params":      "4.2M",
        "speed":       "Balanced",
        "color":       "#3b82f6",
        "desc":        "Compound scaling of depth, width & resolution. Best accuracy per parameter.",
        "input_range": "Internal",
    },
    "Nasnet_mobile": {
        "path":        "classifier_outputs/nasnetmobile.keras",
        "preprocess":  applications.nasnet.preprocess_input,
        "params":      "10.8M",
        "speed":       "Accurate",
        "color":       "#8b5cf6",
        "desc":        "Larger EfficientNet with compound scaling for superior feature extraction.",
        "input_range": "Internal",
    },
    "Model1": {
        "path":        "classifier_outputs/model1.keras",
        "preprocess":  applications.efficientnet.preprocess_input,
        "params":      "N/A",
        "speed":       "Custom",
        "color":       "#f59e0b",
        "desc":        "Custom trained model. Uses EfficientNet-style preprocessing pipeline.",
        "input_range": "Internal",
    },
}

# ── Lazy Model Cache ───────────────────────────────────────────────────────────
_model_cache: dict = {}

def load_model(name: str):
    """Load and cache a .keras model by registry name. Returns None if not found."""
    if name in _model_cache:
        return _model_cache[name]
    cfg  = MODEL_CONFIG.get(name)
    if not cfg:
        return None
    path = cfg["path"]
    if not os.path.exists(path):
        return None
    try:
        model = keras.models.load_model(path)
        _model_cache[name] = model
        return model
    except Exception:
        return None

def get_model_status() -> dict:
    """Return availability status for all models."""
    status = {}
    for name, cfg in MODEL_CONFIG.items():
        status[name] = os.path.exists(cfg["path"])
    return status

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_image(pil_img: Image.Image, model_name: str) -> np.ndarray:
    """Resize, cast to float32 [0,255], apply model-specific preprocess_input."""
    img  = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr  = np.array(img, dtype=np.float32)          # [0, 255] — NO /255
    arr  = np.expand_dims(arr, axis=0)               # (1, 224, 224, 3)
    fn   = MODEL_CONFIG[model_name]["preprocess"]
    return fn(arr)

# ── Prediction ────────────────────────────────────────────────────────────────
def run_prediction(model, tensor: np.ndarray, model_name: str) -> dict:
    """Run inference and return structured result dict."""
    raw_score  = float(model.predict(tensor, verbose=0)[0][0])
    is_real    = raw_score >= 0.5
    label      = "Real" if is_real else "AI-Generated"
    confidence = raw_score if is_real else (1.0 - raw_score)
    uncertain  = confidence < UNCERTAIN_THRESH

    return {
        "label":      label,
        "is_real":    is_real,
        "confidence": round(confidence * 100, 2),
        "raw_score":  round(raw_score, 4),
        "ai_pct":     round((1.0 - raw_score) * 100, 1),
        "real_pct":   round(raw_score * 100, 1),
        "uncertain":  uncertain,
        "model_name": model_name,
        "model_color": MODEL_CONFIG[model_name]["color"],
    }

# ── Helpers ───────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def file_size_str(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes/1024:.1f} KB"
    return f"{size_bytes/1024**2:.1f} MB"

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    """Landing home page."""
    return render_template("home.html")

@app.route("/classify")
def index():
    """Classifier page — upload form."""
    model_status = get_model_status()
    available    = [n for n, ok in model_status.items() if ok]
    return render_template(
        "index.html",
        models=MODEL_CONFIG,
        model_status=model_status,
        available=available,
    )

@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload + model selection, run inference, render result."""
    # ── Validate model selection ──────────────────────────────
    model_name = request.form.get("model_name", "EfficientNetb0")
    if model_name not in MODEL_CONFIG:
        return render_template("index.html",
            error="Invalid model selected.",
            models=MODEL_CONFIG,
            model_status=get_model_status(),
            available=[n for n, ok in get_model_status().items() if ok])

    # ── Validate file ─────────────────────────────────────────
    if "image" not in request.files:
        return render_template("index.html",
            error="No file submitted. Please choose an image.",
            models=MODEL_CONFIG,
            model_status=get_model_status(),
            available=[n for n, ok in get_model_status().items() if ok])

    file = request.files["image"]
    if file.filename == "":
        return render_template("index.html",
            error="No file selected. Please choose an image.",
            models=MODEL_CONFIG,
            model_status=get_model_status(),
            available=[n for n, ok in get_model_status().items() if ok])

    if not allowed_file(file.filename):
        return render_template("index.html",
            error="Unsupported file type. Please upload JPG, PNG, or WEBP.",
            models=MODEL_CONFIG,
            model_status=get_model_status(),
            available=[n for n, ok in get_model_status().items() if ok])

    # ── Read image ────────────────────────────────────────────
    try:
        raw_bytes = file.read()
        pil_img   = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        w, h      = pil_img.size
        size_str  = file_size_str(len(raw_bytes))
        ext       = file.filename.rsplit(".", 1)[1].upper()
    except Exception:
        return render_template("index.html",
            error="Could not read image. The file may be corrupt.",
            models=MODEL_CONFIG,
            model_status=get_model_status(),
            available=[n for n, ok in get_model_status().items() if ok])

    # ── Load model ────────────────────────────────────────────
    model = load_model(model_name)
    if model is None:
        return render_template("index.html",
            error=f"Model '{model_name}' not found. Place the .keras file in classifier_outputs/",
            models=MODEL_CONFIG,
            model_status=get_model_status(),
            available=[n for n, ok in get_model_status().items() if ok])

    # ── Inference ─────────────────────────────────────────────
    try:
        tensor = preprocess_image(pil_img, model_name)
        result = run_prediction(model, tensor, model_name)
    except Exception as e:
        return render_template("index.html",
            error=f"Inference failed: {str(e)}",
            models=MODEL_CONFIG,
            model_status=get_model_status(),
            available=[n for n, ok in get_model_status().items() if ok])

    # ── Encode image for display (base64) ─────────────────────
    import base64
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return render_template(
        "result.html",
        result=result,
        img_b64=img_b64,
        filename=file.filename,
        img_dims=f"{w}×{h}px",
        img_size=size_str,
        img_format=ext,
        models=MODEL_CONFIG,
    )

@app.route("/about")
def about():
    return render_template("about.html", models=MODEL_CONFIG, model_status=get_model_status())

@app.route("/api/models")
def api_models():
    """JSON endpoint — model availability."""
    return jsonify({
        name: {"available": os.path.exists(cfg["path"]), "params": cfg["params"], "speed": cfg["speed"]}
        for name, cfg in MODEL_CONFIG.items()
    })

@app.errorhandler(413)
def too_large(e):
    return render_template("index.html",
        error="File too large. Maximum upload size is 10 MB.",
        models=MODEL_CONFIG,
        model_status=get_model_status(),
        available=[n for n, ok in get_model_status().items() if ok]), 413

@app.errorhandler(404)
def not_found(e):
    return redirect(url_for("home"))

# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("classifier_outputs", exist_ok=True)
    app.run(host="0.0.0.0", port=7860)
