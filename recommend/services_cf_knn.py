# recommend/services_cf_knn.py
from __future__ import annotations
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from django.db.models import Avg, Count

from .models import Myrating, Movie

# --------- Hyperparams (you can tweak) ----------
MIN_OVERLAP = 3        # min number of common users for an item–item similarity to count
BLEND_POP = 0.30       # 0..1; final_score = (1-BLEND_POP)*knn_score + BLEND_POP*pop_score
LIKE_THRESHOLD = 4     # rating >= 4 counts as a "like" (1), else 0

# --------- Cached state (rebuilt lazily) ----------
_item_ids: List[int] = []
_uid_to_row: dict = {}
_mid_to_col: dict = {}
_R_bin: Optional[sparse.csr_matrix] = None     # binary user-item matrix
_S: Optional[sparse.csr_matrix] = None          # cosine similarity (items x items)
_pop_scores: Optional[np.ndarray] = None        # 0..1 popularity proxy aligned with _item_ids
_built = False

def _fetch_ratings_df() -> pd.DataFrame:
    # Only positive ratings; we binarize with LIKE_THRESHOLD
    df = pd.DataFrame(list(Myrating.objects.values("user_id", "movie_id", "rating")))
    if df.empty:
        return df
    return df

def _compute_popularity(item_ids: List[int]) -> np.ndarray:
    """Return popularity score in [0,1] per item, aligned with item_ids."""
    stats = (Movie.objects
        .filter(id__in=item_ids)
        .annotate(num=Count("myrating"), avg=Avg("myrating__rating"))
        .values("id","avg","num"))

    by_id = {s["id"]: s for s in stats}
    # simple popularity proxy: avg * log(1+num)
    vals = []
    for mid in item_ids:
        s = by_id.get(mid, {"avg": 0.0, "num": 0})
        score = float(s["avg"] or 0.0) * np.log1p(int(s["num"] or 0))
        vals.append(score)
    arr = np.array(vals, dtype=np.float64)
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr

def _ensure_model():
    global _item_ids, _uid_to_row, _mid_to_col, _R_bin, _S, _pop_scores, _built
    if _built:
        return

    df = _fetch_ratings_df()
    if df.empty:
        # nothing to build
        _item_ids, _uid_to_row, _mid_to_col = [], {}, {}
        _R_bin = _S = None
        _pop_scores = None
        _built = True
        return

    # keep only positive ratings then binarize
    df = df[df["rating"] > 0].copy()
    df["like"] = (df["rating"] >= LIKE_THRESHOLD).astype(np.int8)
    df = df[df["like"] == 1]
    if df.empty:
        _item_ids, _uid_to_row, _mid_to_col = [], {}, {}
        _R_bin = _S = None
        _pop_scores = None
        _built = True
        return

    user_ids = sorted(df["user_id"].unique().tolist())
    _item_ids = sorted(df["movie_id"].unique().tolist())
    _uid_to_row = {uid: i for i, uid in enumerate(user_ids)}
    _mid_to_col = {mid: j for j, mid in enumerate(_item_ids)}

    rows = df["user_id"].map(_uid_to_row).to_numpy()
    cols = df["movie_id"].map(_mid_to_col).to_numpy()
    data = np.ones(len(df), dtype=np.float32)
    _R_bin = sparse.csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(_item_ids)))

    # item–item cosine: S = (R^T R) / (||col|| * ||col||)
    X = _R_bin  # (U x I)
    C = (X.T @ X).astype(np.float32)  # co-occurrence counts (I x I)
    col_norms = np.sqrt(C.diagonal()).astype(np.float32)
    denom = col_norms[:, None] * col_norms[None, :]
    denom[denom == 0] = 1.0
    S = C / denom

    # zero diag and apply min-overlap
    S.setdiag(0.0)
    if MIN_OVERLAP > 1:
        mask = C.A < MIN_OVERLAP
        S = S.tolil()
        S[mask] = 0.0
        S = S.tocsr()

    _S = S.tocsr()

    # popularity aligned with item order
    _pop_scores = _compute_popularity(_item_ids)

    _built = True

def _user_likes(user) -> List[int]:
    # movie ids user liked (>= threshold)
    qs = Myrating.objects.filter(user=user, rating__gte=LIKE_THRESHOLD).values_list("movie_id", flat=True)
    return list(qs)

def topn_knn_for_user(user, k: int = 10, min_overlap: int = MIN_OVERLAP, blend_pop: float = BLEND_POP) -> List[int]:
    """
    Return top-N movie_ids for a user using binarized item–item cosine KNN.
    - Only considers items with at least min_overlap common users.
    - Optionally blends in a popularity score.
    """
    global MIN_OVERLAP
    if min_overlap != MIN_OVERLAP:
        # allow runtime override
        MIN_OVERLAP = min_overlap
        # force rebuild next call
        global _built
        _built = False

    _ensure_model()
    if _S is None or not _item_ids:
        return []

    if user is None or not getattr(user, "is_authenticated", False):
        # anonymous -> just return top popularity
        order = np.argsort(-_pop_scores)
        return [_item_ids[i] for i in order[:k]]

    liked = _user_likes(user)
    if not liked:
        # no history -> fall back to popularity
        order = np.argsort(-_pop_scores)
        return [_item_ids[i] for i in order[:k]]

    # build a preference vector over item columns
    cols = [ _mid_to_col[mid] for mid in liked if mid in _mid_to_col ]
    if not cols:
        order = np.argsort(-_pop_scores)
        return [_item_ids[i] for i in order[:k]]

    # score = sum of similarities to liked items
    sim_cols = _S[:, cols]                       # (I x |liked|)
    scores = np.asarray(sim_cols.sum(axis=1)).ravel()  # shape [I]

    # exclude already liked
    for mid in liked:
        j = _mid_to_col.get(mid)
        if j is not None:
            scores[j] = -np.inf

    # blend with popularity (normalized 0..1)
    if blend_pop > 0 and _pop_scores is not None:
        # normalize scores to 0..1 if they are finite
        sc = scores.copy()
        sc[np.isneginf(sc)] = 0.0
        if sc.max() > 0:
            sc = sc / sc.max()
        final = (1.0 - blend_pop) * sc + blend_pop * _pop_scores
    else:
        final = scores

    order = np.argsort(-final)
    top = []
    for idx in order:
        if len(top) >= k:
            break
        if np.isneginf(scores[idx]):  # filtered (already liked)
            continue
        top.append(_item_ids[idx])
    return top
