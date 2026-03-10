from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import Movie, Myrating

# -------- Build a light TF-IDF index once (lazy) --------

_vectorizer: Optional[TfidfVectorizer] = None
_item_matrix = None
_movie_ids: List[int] = []
_corpus_built = False

def _movie_text(m: Movie) -> str:
    parts = [
        getattr(m, "title", "") or "",
        getattr(m, "genre", "") or "",      
        getattr(m, "description", "") or "",
    ]
    return " ".join(p for p in parts if p)

def _ensure_index():
    global _vectorizer, _item_matrix, _movie_ids, _corpus_built
    if _corpus_built:
        return
    movies = list(Movie.objects.all())
    _movie_ids = [m.id for m in movies]
    corpus = [_movie_text(m) for m in movies]
    _vectorizer = TfidfVectorizer(min_df=1, max_features=20000, ngram_range=(1,2))
    _item_matrix = _vectorizer.fit_transform(corpus)  # shape: [n_items, n_terms]
    _corpus_built = True

# -------- Public helpers --------

def has_user_history(user) -> bool:
    if user is None or not getattr(user, "is_authenticated", False):
        return False
    return Myrating.objects.filter(user=user, rating__gt=0).exists()

def topn_content_for_user(user, k: int = 10) -> list[int]:
    """
    Content-based recs using a user TF-IDF profile aggregated from the
    user's rated movies, weighted by (rating - 2.5). If the user has no
    history, falls back to popularity handled by caller.
    Returns a list of movie_ids ordered by score.
    """
    _ensure_index()
    if not has_user_history(user):
        return []

    # Collect user's rated movies
    rows = list(Myrating.objects.filter(user=user).values_list("movie_id", "rating"))
    if not rows:
        return []

    # Build user profile vector
    # map rated movie ids -> rows in the TF-IDF matrix
    id_to_row = {mid: i for i, mid in enumerate(_movie_ids)}
    weights = []
    vectors = []
    for mid, r in rows:
        row = id_to_row.get(mid)
        if row is None:
            continue
        w = float(r) - 2.5  # center ratings so >2.5 contributes positive signal
        if w == 0:
            continue
        weights.append(w)
        vectors.append(_item_matrix[row])
    if not vectors:
        return []

    weights = np.array(weights, dtype=np.float32)
    user_vec = (vectors[0] * weights[0]).copy()
    for v, w in zip(vectors[1:], weights[1:]):
        user_vec += v * w

    # Score all items by cosine similarity
    sims = cosine_similarity(user_vec, _item_matrix).ravel()  # shape [n_items]

    # Remove already-rated items
    rated = {mid for mid, _ in rows}
    idx_sorted = np.argsort(-sims)  # descending
    result_ids = []
    for idx in idx_sorted:
        mid = _movie_ids[idx]
        if mid in rated:
            continue
        result_ids.append(mid)
        if len(result_ids) >= k:
            break
    return result_ids
