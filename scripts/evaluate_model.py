import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import os
import pickle
from collections import defaultdict
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(BASE_DIR, 'ml_models')

# Load mappings
with open(os.path.join(models_dir, 'mappings.pkl'), 'rb') as f:
    mappings = pickle.load(f)

user2idx = mappings['user2idx']
movie2idx = mappings['movie2idx']
global_mean = mappings.get('global_mean', 3.5)
embedding_size = mappings.get('embedding_size', 50)

# Load pre-split data
print("Loading pre-split train/test data...")
train_df = pd.read_csv(os.path.join(models_dir, 'train_data.csv'))
test_df = pd.read_csv(os.path.join(models_dir, 'test_data.csv'))

# Model - must match train script exactly
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

# Load trained model
model = RecommenderNet(len(user2idx), len(movie2idx), global_mean, embedding_size)
model.load_state_dict(torch.load(os.path.join(models_dir, 'recommender_weights.pth')))
model.eval()

# Inference
print("Calculating predictions on test set...")
test_users = torch.tensor(test_df['user_idx'].values, dtype=torch.long)
test_movies = torch.tensor(test_df['movie_idx'].values, dtype=torch.long)
actual_ratings = test_df['rating'].values

with torch.no_grad():
    predicted_ratings = model(test_users, test_movies).numpy()

# Clip to valid range
predicted_ratings = np.clip(predicted_ratings, 1.0, 5.0)

# Metrics
mae = mean_absolute_error(actual_ratings, predicted_ratings)
rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))

print("-" * 30)
print("PYTORCH MODEL EVALUATION METRICS")
print("-" * 30)
print(f"Mean Absolute Error (MAE): {mae:.4f} stars")
print(f"Root Mean Square Error (RMSE): {rmse:.4f} stars")
print("-" * 30)
print("Note: A random guesser typically scores RMSE ~1.5-2.0")
print("An excellent production model scores RMSE between 0.85-0.95")
print("-" * 30)

# Top-K Evaluation
print("Calculating Precision, Recall, and Hit Rate using Candidate Sampling...")

def evaluate_top_k_sampled(model, train_df, test_df, k=12, num_negatives=99):
    relevant_items = defaultdict(list)
    for _, row in test_df[test_df['rating'] >= 4.0].iterrows():
        relevant_items[row['user_idx']].append(row['movie_idx'])

    train_items = defaultdict(set)
    for _, row in train_df.iterrows():
        train_items[row['user_idx']].add(row['movie_idx'])

    all_movie_indices = list(movie2idx.values())

    precisions, recalls, hit_rates = [], [], []

    model.eval()
    with torch.no_grad():
        for user_idx, true_relevant_list in relevant_items.items():
            if not true_relevant_list:
                continue

            # Track seen set so same negative is never added twice
            seen = train_items[user_idx] | set(true_relevant_list)
            negatives = []
            while len(negatives) < num_negatives:
                candidate = random.choice(all_movie_indices)
                if candidate not in seen:
                    negatives.append(candidate)
                    seen.add(candidate)  # prevents duplicates

            candidates = true_relevant_list + negatives

            user_tensor = torch.tensor([user_idx] * len(candidates), dtype=torch.long)
            movie_tensor = torch.tensor(candidates, dtype=torch.long)

            preds = model(user_tensor, movie_tensor).numpy()
            top_k_indices = np.argsort(preds)[-k:][::-1]
            top_k_movies = [candidates[i] for i in top_k_indices]

            hits = sum(1 for m in top_k_movies if m in true_relevant_list)

            hit_rates.append(1 if hits > 0 else 0)
            precisions.append(hits / k)
            recalls.append(hits / len(true_relevant_list))

    return np.mean(precisions), np.mean(recalls), np.mean(hit_rates)

p, r, hr = evaluate_top_k_sampled(model, train_df, test_df, k=12)

print(f"Precision@12: {p*100:.2f}%")
print(f"Recall@12:    {r*100:.2f}%")
print(f"Hit Rate@12:  {hr*100:.2f}%")
print("-" * 30)