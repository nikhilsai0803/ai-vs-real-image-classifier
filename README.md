# 🧿 AI vs Real Image Classifier

> Detect AI-generated artwork from real photographs using transfer learning — MobileNetV2, EfficientNetB0, NASNetMobile, and Model1, fine-tuned with a two-phase training strategy.

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=flat&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 📸 Demo

Upload any image and the app will tell you whether it's AI-generated or a real photograph — with a confidence score and score breakdown bar.

**🚀 Live demo →** [huggingface.co/spaces/sharmasai12/AI_vs_REAL](https://huggingface.co/spaces/sharmasai12/AI_vs_REAL)

---

## 🧠 How It Works

Four pretrained ImageNet models are fine-tuned in **two phases** on a dataset of ~4,700 images:

| Phase | What happens |
|-------|-------------|
| **Phase 1** | Base model frozen. Only the classification head trains (10 epochs, lr=1e-3) |
| **Phase 2** | Last 30 layers of base unfrozen. Fine-tune gently (10 epochs, lr=1e-5) |

Each model uses its own `preprocess_input` function to ensure pixel values arrive in the correct range that the ImageNet weights expect.

### Models

| Model | Params | Speed |
|-------|--------|-------|
| MobileNetV2 | 2.4M | Fastest |
| EfficientNetB0 | 4.2M | Balanced |
| NASNetMobile | 4.4M | Most robust |
| Model1 | — | Custom |

### Dataset

[tristanzhang32/ai-generated-images-vs-real-images](https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images) — ~2,300 AI-generated artworks + ~2,400 real photographs.

---

## 🗂️ Project Structure

```
AI_vs_REAL/
│
├── app.py                             ← Flask web app
├── Dockerfile                         ← Docker config for HF Spaces
├── requirements.txt                   ← Python dependencies
├── .gitattributes                     ← Git LFS tracking
├── .gitignore
├── README.md
├── AI vs REAL Image_Classifier.ipynb  ← Training notebook
│
├── templates/                         ← HTML templates
│   ├── index.html
│   ├── home.html
│   ├── about.html
│   └── result.html
│
├── static/                            ← CSS & JS
│   ├── css/style.css
│   └── js/main.js
│
└── classifier_outputs/                ← Model files (tracked via Git LFS)
    ├── mobilenetv2.keras
    ├── efficientnetb0.keras
    ├── nasnetmobile.keras
    └── model1.keras
```

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/nikhilsai0803/AI_vs_REAL.git
cd AI_vs_REAL
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the app
```bash
python app.py
```

Visit `http://localhost:7860` in your browser.

---

## ☁️ Deployed on Hugging Face Spaces

This app is live on **Hugging Face Spaces** using Docker.

| Detail | Info |
|--------|------|
| 🐳 Deployment | Docker-based |
| ⚡ Server | Gunicorn on port `7860` |
| 📦 Models | Tracked with Git LFS (~218MB) |
| 🔗 Live URL | [sharmasai12/AI_vs_REAL](https://huggingface.co/spaces/sharmasai12/AI_vs_REAL) |

---

## 📊 Results

| Model | Test Accuracy |
|-------|-------------|
| MobileNetV2 | ~78–82% |
| EfficientNetB0 | ~75–80% |
| NASNetMobile | ~76–81% |
| Model1 | ~90-93% |

---

## 🛠️ Key Technical Decisions

- **No `/255` in preprocessing** — each model's `preprocess_input()` handles normalisation. Applying `/255` before calling it causes ~49% accuracy (random guessing).
- **Partial fine-tuning** — only the last 30 layers of the base are unfrozen in Phase 2. Full unfreezing destroys pretrained weights.
- **Fresh callbacks per phase** — `EarlyStopping` is re-instantiated before Phase 2 so stale patience counters don't interfere.
- **Docker deployment** — ensures consistent environment on HF Spaces.
- **Git LFS** — handles large `.keras` model files cleanly.

---

## 📝 License

MIT — free to use, modify, and distribute.

---

*Built with TensorFlow, Flask, Docker, and deployed on Hugging Face Spaces.*
