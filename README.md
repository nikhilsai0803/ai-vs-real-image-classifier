# 🧿 AI vs Real Image Classifier

> Detect AI-generated artwork from real photographs using transfer learning — MobileNetV2, EfficientNetB0, and NASNetMobile, fine-tuned with a two-phase training strategy.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR_APP_URL.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 📸 Demo

Upload any image and the app will tell you whether it's AI-generated or a real photograph — with a confidence score and score breakdown bar.

**Live demo →** [your-app-url.streamlit.app](https://YOUR_APP_URL.streamlit.app)

---

## 🧠 How It Works

Three pretrained ImageNet models are fine-tuned in **two phases** on a dataset of ~4,700 images:

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

### Dataset

[tristanzhang32/ai-generated-images-vs-real-images](https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images) — ~2,300 AI-generated artworks + ~2,400 real photographs.

---

## 🗂️ Project Structure

```
ai-vs-real-classifier/
│
├── app.py                        ← Streamlit web app
├── AI_Image_Classifier.ipynb     ← Training notebook
├── requirements.txt              ← Python dependencies
├── packages.txt                  ← System dependencies (Streamlit Cloud)
├── .gitignore
├── README.md
│
└── classifier_outputs/           ← Generated after training
    ├── mobilenetv2.keras
    ├── efficientnetb0.keras
    ├── nasnetmobile.keras
    ├── best_model.keras
    ├── 01_dataset_overview.png
    ├── 02_sample_images.png
    ├── 03_image_properties.png
    ├── 04_training_curves.png
    ├── 05_training_time.png
    ├── 06_confusion_matrices.png
    └── 07_model_comparison.png
```

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/ai-vs-real-classifier.git
cd ai-vs-real-classifier
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

### 4. Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images) and extract so you have:
```
archive/
├── AiArtData/
└── RealArt/
```
Update `BASE_DIR` in the notebook to point to your `archive/` folder.

### 5. Train the models
Open and run `AI_Image_Classifier.ipynb` top to bottom.  
This creates the `classifier_outputs/` folder with all `.keras` model files.

### 6. Launch the app
```bash
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Community Cloud (free)

> ⚠️ The `.keras` model files are large (~50–100MB each). You **cannot** push them to GitHub directly. Use Git LFS or host them externally (see below).

### Option A — Git LFS (recommended for beginners)

```bash
# Install Git LFS (one time)
git lfs install

# Track keras model files
git lfs track "*.keras"

# This creates .gitattributes — commit it
git add .gitattributes
git commit -m "Track keras models with LFS"

# Now add and push your models normally
git add classifier_outputs/*.keras
git commit -m "Add trained models"
git push
```

Then on [share.streamlit.io](https://share.streamlit.io):
1. Connect your GitHub account
2. Select this repo, branch `main`, file `app.py`
3. Click **Deploy**

### Option B — Host models on Google Drive / HuggingFace Hub

If you don't want to use LFS, you can upload models to HuggingFace Hub and download them on app startup. Ask for this snippet if needed.

---

## 📊 Results

After training with the fixed preprocessing pipeline:

| Model | Test Accuracy | Test Loss |
|-------|-------------|-----------|
| MobileNetV2 | ~78–82% | — |
| EfficientNetB0 | ~75–80% | — |
| NASNetMobile | ~76–81% | — |

*(Exact numbers depend on your run — update this table after training)*

---

## 🛠️ Key Technical Decisions

- **No `/255` in preprocessing** — each model's `preprocess_input()` handles normalisation. Applying `/255` before calling it was the single biggest bug causing ~49% accuracy (random guessing).
- **Partial fine-tuning** — only the last 30 layers of the base are unfrozen in Phase 2. Full unfreezing destroys pretrained weights.
- **Fresh callbacks per phase** — `EarlyStopping` is re-instantiated before Phase 2 so stale patience counters don't interfere.
- **Image validation at startup** — corrupt/tiny images are detected and removed before training begins.

---

## 📝 License

MIT — free to use, modify, and distribute.

---

*Built with TensorFlow, Streamlit, and a lot of debugging.*
