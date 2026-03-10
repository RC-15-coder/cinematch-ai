import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(BASE_DIR, 'ml_models')
os.makedirs(models_dir, exist_ok=True)

# Load Data
ratings_path = os.path.join(BASE_DIR, 'ml_data', 'ml-latest-small', 'ratings.csv')
df = pd.read_csv(ratings_path)

# Split FIRST before any filtering or statistics
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Calculate popularity filter using ONLY train data
movie_counts = train_df['movieId'].value_counts()
popular_movies = movie_counts[movie_counts >= 25].index
train_df = train_df[train_df['movieId'].isin(popular_movies)]
test_df = test_df[test_df['movieId'].isin(popular_movies)]

global_mean = train_df['rating'].mean()
print(f"Global Mean Rating (train only): {global_mean:.4f}")

# Encode IDs using ONLY train data
user_ids = train_df['userId'].unique()
movie_ids = train_df['movieId'].unique()
user2idx = {o: i for i, o in enumerate(user_ids)}
movie2idx = {o: i for i, o in enumerate(movie_ids)}

# Filter test to only include users/movies seen in training
test_df = test_df[test_df['userId'].isin(user2idx)]
test_df = test_df[test_df['movieId'].isin(movie2idx)]

train_df['user_idx'] = train_df['userId'].map(user2idx)
train_df['movie_idx'] = train_df['movieId'].map(movie2idx)
test_df['user_idx'] = test_df['userId'].map(user2idx)
test_df['movie_idx'] = test_df['movieId'].map(movie2idx)

# Save splits
train_df.to_csv(os.path.join(models_dir, 'train_data.csv'), index=False)
test_df.to_csv(os.path.join(models_dir, 'test_data.csv'), index=False)
print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

# Dataset
class MovieDataset(Dataset):
    def __init__(self, users, movies, ratings):
        self.users = torch.tensor(users, dtype=torch.long)
        self.movies = torch.tensor(movies, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

dataset = MovieDataset(
    train_df['user_idx'].values,
    train_df['movie_idx'].values,
    train_df['rating'].values
)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

# Model
class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_movies, global_mean, embedding_size=50):
        super(RecommenderNet, self).__init__()
        self.global_mean = global_mean
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)

        # Small initialization prevents exploding dot products
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

    def forward(self, user_idx, movie_idx):
        user_vec = self.user_embedding(user_idx)
        movie_vec = self.movie_embedding(movie_idx)
        u_bias = self.user_bias(user_idx).squeeze(-1)
        m_bias = self.movie_bias(movie_idx).squeeze(-1)
        dot = (user_vec * movie_vec).sum(1)
        return self.global_mean + u_bias + m_bias + dot

model = RecommenderNet(len(user2idx), len(movie2idx), global_mean, embedding_size=50)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-2)

# Training
print("\nTraining...")
epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for users, movies, ratings in dataloader:
        optimizer.zero_grad()
        preds = model(users, movies)
        loss = criterion(preds, ratings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")

# Save
torch.save(model.state_dict(), os.path.join(models_dir, 'recommender_weights.pth'))
with open(os.path.join(models_dir, 'mappings.pkl'), 'wb') as f:
    pickle.dump({
        'user2idx': user2idx,
        'movie2idx': movie2idx,
        'global_mean': global_mean,
        'embedding_size': 50
    }, f)

print("\nSUCCESS: Model saved!")