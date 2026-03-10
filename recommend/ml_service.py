import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from django.db.models import Avg, Count
from .models import Movie, Myrating

# match train_recommender.py exactly
class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_movies, global_mean, embedding_size=50):
        super(RecommenderNet, self).__init__()
        self.global_mean = global_mean
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)

    def forward(self, user_idx, movie_idx):
        user_vec = self.user_embedding(user_idx)
        movie_vec = self.movie_embedding(movie_idx)
        u_bias = self.user_bias(user_idx).squeeze(-1)
        m_bias = self.movie_bias(movie_idx).squeeze(-1)
        dot = (user_vec * movie_vec).sum(1)
        return self.global_mean + u_bias + m_bias + dot

# Load once when server starts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'ml_models')

with open(os.path.join(MODEL_DIR, 'mappings.pkl'), 'rb') as f:
    mappings = pickle.load(f)

user2idx = mappings['user2idx']
movie2idx = mappings['movie2idx']
idx2movie = {v: k for k, v in movie2idx.items()}
global_mean = mappings.get('global_mean', 3.5)
embedding_size = mappings.get('embedding_size', 50)

num_users = len(user2idx)
num_movies = len(movie2idx)

model = RecommenderNet(num_users, num_movies, global_mean, embedding_size)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'recommender_weights.pth')))
model.eval()

def get_pytorch_recommendations(liked_movie_ids, top_k=12):
    if not liked_movie_ids:
        return []

    valid_movie_indices = [movie2idx[m_id] for m_id in liked_movie_ids if m_id in movie2idx]

    if not valid_movie_indices:
        return []

    movie_tensor = torch.tensor(valid_movie_indices, dtype=torch.long)
    with torch.no_grad():
        liked_embeddings = model.movie_embedding(movie_tensor)
        user_profile = liked_embeddings.mean(dim=0, keepdim=True)
        all_embeddings = model.movie_embedding.weight
        similarities = F.cosine_similarity(user_profile, all_embeddings)
        top_scores, top_indices = torch.topk(similarities, top_k + len(valid_movie_indices))

    recommended_ids = []
    for idx in top_indices:
        if idx.item() not in valid_movie_indices:
            recommended_ids.append(idx2movie[idx.item()])
            if len(recommended_ids) == top_k:
                break

    return recommended_ids

def get_recommendations_for_user(user_id: int | None, k: int = 10):
    qs = Movie.objects.all()

    if user_id:
        try:
            qs = qs.exclude(myrating__user_id=user_id)
        except Exception:
            qs = qs.exclude(myrating_set__user_id=user_id)

    try:
        qs = qs.annotate(
            num=Count("myrating"),
            avg=Avg("myrating__rating"),
        )
    except Exception:
        qs = qs.annotate(
            num=Count("myrating_set"),
            avg=Avg("myrating_set__rating"),
        )

    qs = qs.filter(num__gte=1).order_by("-avg", "-num")[:k]

    data = []
    for m in qs:
        data.append({
            "id": m.id,
            "title": getattr(m, "title", str(m)),
            "score": float(getattr(m, "avg", 0.0) or 0.0),
        })
    return data