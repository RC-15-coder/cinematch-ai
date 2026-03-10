# 🎬 CineMatch AI — Movie Recommender System

[![Live Demo](https://img.shields.io/badge/Live_Demo-Hugging_Face-yellow.svg)](https://rc-15-cinematch-ai.hf.space)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ML_Pipeline-EE4C2C.svg)](https://pytorch.org/)
[![Django](https://img.shields.io/badge/Django-Web_Framework-092E20.svg)](https://www.djangoproject.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)

An end-to-end machine learning recommender system built with **PyTorch** and **Django**. Features a custom-trained Matrix Factorization model using the MovieLens dataset, rigorously evaluated with production-grade ranking metrics, and deployed to the cloud via Docker on Hugging Face Spaces.

**[🚀 Try the Live App Here](https://rc-15-cinematch-ai.hf.space)**

---

## 🧠 Machine Learning Architecture

The recommendation engine uses **Matrix Factorization with learned embeddings** — the same foundational approach used by Netflix and Spotify.

- **Algorithm:** PyTorch Matrix Factorization with User/Movie Embeddings + Bias terms + Global Mean Normalization
- **Dataset:** `ml-latest-small` MovieLens dataset (~100K ratings), filtered to movies with 25+ ratings for embedding stability
- **Inference:** Cosine Similarity on learned movie embeddings to build a user taste profile at runtime
- **Cold Start Handling:** Users must rate a minimum threshold of movies before recommendations activate

### 📊 Evaluation Metrics

| Metric | Score | Benchmark |
|--------|-------|-----------|
| RMSE | **0.9213** | Production range: 0.85–0.95 |
| MAE | **0.7231** | — |
| Hit Rate@12 | **81.71%** | Typical MF range: 30–50% |
| Precision@12 | **24.66%** | — |
| Recall@12 | **33.49%** | — |

> **Note:** Evaluation used Candidate Sampling (99 negatives per user) with a strict train/test split to prevent data leakage. Global Mean was calculated exclusively on training data.

---

## 🛠️ Tech Stack

- **Machine Learning:** PyTorch, NumPy, Pandas, Scikit-Learn
- **Backend:** Django 4.2, SQLite, Gunicorn
- **Frontend:** HTML5, CSS3, JavaScript (jQuery), Bootstrap, Django Templates
- **Deployment:** Docker, Hugging Face Spaces, WhiteNoise

---

## 📂 Project Structure
```text
├── Dockerfile                      # Production containerization
├── requirements.txt                # All dependencies
├── manage.py                       # Django entry point
├── movie_recommender/              # Django project settings & routing
├── recommend/                      # App logic, models, views, ML service
│   └── ml_service.py               # PyTorch inference + cosine similarity
├── scripts/
│   ├── train_recommender.py        # PyTorch training pipeline
│   ├── evaluate_model.py           # RMSE, MAE, HitRate, Precision, Recall
│   ├── eval.py                     # Supplementary evaluation script
│   └── eval_cosine.py              # Cosine similarity vector evaluation
└── ml_models/
    ├── recommender_weights.pth     # Trained PyTorch model weights
    └── mappings.pkl                # User/Movie ID mappings + global mean
```

---

## 💻 Running Locally
```bash
# 1. Clone and setup
git clone https://github.com/RC-15-dev/cinematch-ai.git
cd cinematch-ai
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 2. Download dataset
# Download ml-latest-small.zip from https://grouplens.org/datasets/movielens/
# Extract into ml_data/ml-latest-small/

# 3. Train the model
python scripts/train_recommender.py

# 4. Evaluate
python scripts/evaluate_model.py

# 5. Run the server
python manage.py runserver
```

---

## 🏗️ Architecture Decisions & Limitations

**Why Cosine Similarity instead of raw predictions?**
The model uses unbounded dot products during training for stable gradient flow. At inference time, Cosine Similarity on the learned embeddings provides superior ranking quality — this is why Hit Rate (81.71%) is strong despite the model not being optimized purely for rating prediction.

**Known limitation:**
On smaller genre clusters such as Romance, pure Collaborative Filtering occasionally surfaces cross-genre recommendations due to viewing pattern overlap in the dataset. A Content-Based filtering layer using genre metadata would address this in a production system — a deliberate architectural trade-off given the dataset size.

---

## 🚀 Deployment

Containerized with Docker and deployed on **Hugging Face Spaces**.

> ⚠️ The free tier sleeps after 48 hours of inactivity. First load after sleep takes ~30–60 seconds.