# scripts/eval.py
import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "movie_recommender.settings")
import django
django.setup()

from collections import defaultdict
import numpy as np
import pandas as pd
from django.db.models import Avg, Count
from recommend.models import Myrating, Movie

K = 10

def precision_recall_at_k(recommended, relevant, k=10):
    if not recommended:
        return 0.0, 0.0
    rec_k = recommended[:k]
    rel_set = set(relevant)
    hit = sum(1 for x in rec_k if x in rel_set)
    prec = hit / float(k)
    rec = hit / float(len(rel_set)) if rel_set else 0.0
    return prec, rec

def ndcg_at_k(recommended, relevant, k=10):
    rel_set = set(relevant)
    dcg = 0.0
    for i, mid in enumerate(recommended[:k], start=1):
        if mid in rel_set:
            dcg += 1.0 / np.log2(i + 1)
    ideal_hits = min(len(rel_set), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0

def build_popularity():
    pop = (Movie.objects
           .annotate(num=Count("myrating"), avg=Avg("myrating__rating"))
           .filter(num__gte=1)
           .values_list("id", "avg", "num"))
    pop = sorted(pop, key=lambda t: (t[1] or 0.0, t[2]), reverse=True)
    return [mid for (mid, _, _) in pop]

def item_item_cf_train(df_train):
    # user-item with only positive ratings
    ui = df_train.pivot_table(index="user_id", columns="movie_id", values="rating")
    # mean-center per user (subtract row means)
    ui = ui.sub(ui.mean(axis=1), axis=0).fillna(0.0)
    if ui.shape[1] == 0:
        return pd.DataFrame()
    corr = ui.corr(method="pearson")  # item-item correlation
    return corr

def item_item_recommend(corr, user_hist_centered, exclude_ids, k=10):
    """user_hist_centered: list[(movie_id, centered_rating)] with centered_rating != 0"""
    if corr.empty:
        return []
    scores = pd.Series(dtype=float)
    for mid, cr in user_hist_centered:
        if mid not in corr.columns:
            continue
        scores = scores.add(corr[mid] * cr, fill_value=0.0)
    if scores.empty:
        return []
    scores = scores.sort_values(ascending=False)
    out = [mid for mid in scores.index if mid not in exclude_ids]
    return out[:k]

def main():
    # Pull ratings and use id as time proxy; filter out non-positive ratings
    df_all = (pd.DataFrame(list(Myrating.objects.values("id","user_id","movie_id","rating")))
              .sort_values("id"))
    df_all = df_all[df_all["rating"] > 0]

    if df_all.empty:
        print("No positive ratings found; add some and rerun.")
        return

    # Per-user chronological split: first 80% of each user's ratings -> train, rest -> test
    train_rows, test_rows = [], []
    for uid, grp in df_all.groupby("user_id"):
        n = len(grp)
        if n == 1:
            # not enough to split—put into train
            train_rows.append(grp)
            continue
        cut = int(n * 0.8)
        if cut < 1:
            cut = 1
        train_rows.append(grp.iloc[:cut])
        test_rows.append(grp.iloc[cut:])
    train = pd.concat(train_rows, ignore_index=True)
    test  = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame(columns=df_all.columns)

    # Keep only users with enough history for evaluation: >=3 train & >=1 test
    train_counts = train.groupby("user_id").size()
    test_counts  = test.groupby("user_id").size()
    eval_users = sorted(set(test_counts[test_counts >= 1].index) & set(train_counts[train_counts >= 3].index))
    if not eval_users:
        print("No users with >=3 train ratings and >=1 test rating. Add more ratings and rerun.")
        return

    # Build models
    pop_list = build_popularity()
    corr = item_item_cf_train(train)

    # Precompute user mean for centering in inference
    user_means = train.groupby("user_id")["rating"].mean().to_dict()

    rows = []
    for uid in eval_users:
        rel = list(test.loc[test["user_id"] == uid, "movie_id"].values)
        hist = train.loc[train["user_id"] == uid, ["movie_id","rating"]]

        # exclude already seen
        exclude = set(hist["movie_id"].values)

        # Popularity@K (filtered)
        pop_rec = [mid for mid in pop_list if mid not in exclude][:K]

        # CF@K (center user ratings same as training centering)
        mu = user_means.get(uid, 0.0)
        user_hist_centered = [(int(r.movie_id), float(r.rating) - mu) for r in hist.itertuples(index=False)]
        # drop zeros after centering
        user_hist_centered = [(mid, cr) for mid, cr in user_hist_centered if abs(cr) > 1e-6]
        cf_rec = item_item_recommend(corr, user_hist_centered, exclude, k=K)

        p_pop, r_pop = precision_recall_at_k(pop_rec, rel, K); n_pop = ndcg_at_k(pop_rec, rel, K)
        p_cf,  r_cf  = precision_recall_at_k(cf_rec,  rel, K); n_cf  = ndcg_at_k(cf_rec,  rel, K)

        rows.append({
            "user_id": uid,
            "prec_pop": p_pop, "rec_pop": r_pop, "ndcg_pop": n_pop,
            "prec_cf":  p_cf,  "rec_cf":  r_cf,  "ndcg_cf":  n_cf,
        })

    res = pd.DataFrame(rows)
    summary = res[["prec_pop","rec_pop","ndcg_pop","prec_cf","rec_cf","ndcg_cf"]].mean().to_frame("mean").round(4)
    print("\n=== Offline Evaluation @K=10 (per-user time split 80/20; users with ≥3 train & ≥1 test) ===")
    print(summary)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(os.path.join(out_dir, "offline_eval_summary.csv"))

if __name__ == "__main__":
    main()
