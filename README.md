# RareEyeAI — Few-Shot Learning for Rare Ophthalmic Disease Detection

A complete, runnable **Flask** web application (HTML + CSS + Bootstrap 5 + vanilla JS
frontend, Python/Flask backend, SQL database) that demonstrates a **Prototypical
Network** (few-shot learning) pipeline for classifying retinal fundus images into
rare eye disease categories, with **Grad-CAM** visual explanations, user accounts,
history, and downloadable PDF reports.

## Features

- 🔐 **Authentication** — register / login / logout (Flask-Login, hashed passwords)
- 📤 **Interactive upload UI** — drag & drop, live image preview, loading overlay
- 🧠 **Few-shot classification** — a CNN encoder embeds the query image and classifies
  it by nearest **class prototype** (mean embedding of a K-shot support set), exactly
  following Snell et al.'s *Prototypical Networks for Few-shot Learning* (2017)
- 🔥 **Grad-CAM explanations** — real gradient-based attention heatmaps computed by
  backpropagating the similarity score to the predicted prototype through the CNN
- 📊 **Dashboard** — stats, prediction distribution, recent analyses
- 🕘 **History** — table of all past analyses with view/delete
- 📄 **PDF reports** — one-click downloadable prediction report (ReportLab)
- 🗄️ **Database** — SQLAlchemy models for `User` and `Prediction`; SQLite by default,
  one env var away from PostgreSQL
- 🎨 **Responsive Bootstrap 5 UI** with a custom design system (custom CSS, icons, fonts)

## 🔌 First-run internet requirement

On the very first run, `ml/network.py` downloads a ~44MB **ImageNet-pretrained
ResNet-18** checkpoint via `torchvision` (cached under
`~/.cache/torch/hub/checkpoints/` afterwards, no internet needed on later
runs). This pretrained backbone is what makes predictions differentiate
meaningfully between different uploaded images — a randomly-initialized CNN
(no pretraining) produces near-random embeddings, so every image ends up
almost equidistant from all class prototypes and confidence collapses to
roughly `1/N` for every class (e.g. ~20% for 5 classes) no matter what you
upload. If the download fails (offline environment), the app automatically
falls back to a randomly-initialized backbone with a console warning — the
app still runs, but predictions will look close to uniform until you re-run
with internet access.

## ⚠️ Important note on the model & data

There is no publicly bundled, licensed dataset of rare ophthalmic disease fundus
photographs available in this environment. To make the app **fully runnable
end-to-end out of the box**, it ships with:

1. A **procedurally generated synthetic support set** (`ml/demo_support_set.py`)
   that creates stylised, fundus-like images with class-specific visual signatures
   for 5 demo "rare disease" classes.
2. A small **CNN embedding network** (`ml/network.py`) with a fixed random seed
   (not trained on real labelled data).

This means predictions and heatmaps are for **demonstrating the engineering
pipeline** (Prototypical Network inference + Grad-CAM + full-stack app), **not**
for real diagnosis. See "Training on a real dataset" below to make it clinically
meaningful.

## Project structure

```
ophthalmic_app/
├── app.py                  # Flask app factory + all routes
├── config.py                # App configuration (DB URI, folders, few-shot settings)
├── extensions.py             # db, login_manager singletons
├── models.py                  # User, Prediction SQLAlchemy models
├── forms.py                    # WTForms: Register, Login, Upload
├── report.py                    # PDF report generator (ReportLab)
├── ml/
│   ├── network.py                # EmbeddingNet CNN (Prototypical Network backbone)
│   ├── prototypical.py            # Build prototypes + classify query image
│   ├── gradcam.py                  # Grad-CAM over the embedding network
│   └── demo_support_set.py          # Synthetic support-set image generator
├── templates/                        # Jinja2 + Bootstrap 5 templates
│   ├── base.html, index.html, login.html, register.html,
│   │   dashboard.html, upload.html, result.html, history.html,
│   │   support_set.html, error.html
├── static/
│   ├── css/style.css, js/main.js
│   ├── uploads/, heatmaps/, reports/, support_set/   (created at runtime)
├── requirements.txt
└── README.md
```

## Setup & Run (local)

```bash
cd ophthalmic_app
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python app.py
# Visit http://localhost:5000
```

On first run the app automatically:
- creates the SQLite database (`instance/app.db`) and tables
- generates the synthetic support set images under `static/support_set/`
- pre-computes the class prototypes

## Using PostgreSQL instead of SQLite

```bash
export DATABASE_URL="postgresql://username:password@localhost:5432/ophthalmic_db"
python app.py
```
(add `psycopg2-binary` is already in `requirements.txt`)

## Deployment (production)

```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```

Put this behind Nginx / a platform such as Render, Railway, Fly.io, or a Docker
container. Example minimal `Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV SECRET_KEY=change-me DATABASE_URL=sqlite:////app/instance/app.db
EXPOSE 8000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "app:app"]
```

## How the Prototypical Network works here

1. **Support set**: for each of the `N` rare-disease classes, `K` example images are
   embedded by the CNN encoder (`EmbeddingNet`) and averaged into one **prototype**
   vector per class (`ml/prototypical.py::build_prototypes`).
2. **Query classification**: an uploaded image is embedded the same way, then
   classified by computing the (negative) squared Euclidean distance to every
   prototype and taking a softmax over those distances — the class with the
   smallest distance (highest probability) wins.
3. **Grad-CAM**: the similarity score of the query to its predicted prototype is
   backpropagated through the CNN; gradients at the last convolutional layer are
   global-average-pooled into channel weights, combined with the activations, and
   ReLU'd to produce a coarse localization heatmap, which is upsampled and blended
   over the original image (`ml/gradcam.py`).

## Training on a real dataset (going from demo → production)

1. Collect / obtain a licensed, IRB-approved set of fundus images per rare disease
   class (a handful of confirmed cases per class is enough for K-shot learning).
2. Replace the contents of `static/support_set/<ClassName>/` with real images (or
   point `ensure_support_set` at a different, pre-populated folder and delete the
   synthetic generator call).
3. (Recommended) Train the `EmbeddingNet` with an episodic few-shot training loop
   (sample N-way K-shot episodes from a larger labelled dataset, minimize
   prototypical-network cross-entropy loss) instead of using the randomly
   initialized network, then load the trained weights in `ml/network.py`.
4. Update `RARE_DISEASE_CLASSES` in `ml/demo_support_set.py` to your real class
   names.

## Tech stack

Frontend: HTML5, CSS3, Bootstrap 5, Bootstrap Icons, vanilla JavaScript (drag & drop,
animated bars, confidence ring, spinner overlay)
Backend: Python, Flask, Flask-Login, Flask-WTF, SQLAlchemy
ML: PyTorch / TorchVision (CNN encoder, Prototypical Network, Grad-CAM), Pillow, NumPy
Reports: ReportLab (PDF generation)
Database: SQLite (default) / PostgreSQL (via `DATABASE_URL`)

## Disclaimer

This is a research / educational demonstration system. It is **not** a certified
medical device and must not be used for real clinical diagnosis. All predictions
must be reviewed by a qualified ophthalmologist.