# scripts/eval_cosine.py
import os, sys
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path: sys.path.insert(0, BASE)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "movie_recommender.settings")
import django; django.setup()

import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import sparse
from django.db.models import Avg, Count
from recommend.models import Myrating, Movie

K = 10
LIKE_THRESHOLD = 4
MIN_OVERLAP = 3
BLEND_POP = 0.30  # set to 0.0 if you want pure cosine

def precision_recall_at_k(recommended, relevant, k=10):
    if not recommended: return 0.0, 0.0
    rec_k = recommended[:k]; rel = set(relevant)
    hit = sum(1 for x in rec_k if x in rel)
    prec = hit / float(k)
    rec = hit / float(len(rel)) if rel else 0.0
    return prec, rec

def ndcg_at_k(recommended, relevant, k=10):
    rel = set(relevant); dcg = 0.0
    for i, mid in enumerate(recommended[:k], start=1):
        if mid in rel: dcg += 1.0/np.log2(i+1)
    ideal = min(len(rel), k)
    idcg = sum(1.0/np.log2(i+1) for i in range(1, ideal+1))
    return dcg/idcg if idcg>0 else 0.0

def build_popularity(item_ids):
    stats = (Movie.objects.filter(id__in=item_ids)
             .annotate(num=Count("myrating"), avg=Avg("myrating__rating"))
             .values("id","avg","num"))
    by_id = {s["id"]: s for s in stats}
    vals = []
    for mid in item_ids:
        s = by_id.get(mid, {"avg":0.0,"num":0})
        score = float(s["avg"] or 0.0) * np.log1p(int(s["num"] or 0))
        vals.append(score)
    arr = np.array(vals, dtype=np.float64)
    if arr.max() > 0: arr = arr/arr.max()
    return arr

def build_knn(train_df):
    # binarize likes
    df = train_df.copy()
    df = df[df["rating"] > 0]
    df["like"] = (df["rating"] >= LIKE_THRESHOLD).astype(np.int8)
    df = df[df["like"] == 1]
    if df.empty:
        return [], {}, sparse.csr_matrix((0,0)), None, None

    users = sorted(df["user_id"].unique().tolist())
    items = sorted(df["movie_id"].unique().tolist())
    uid2row = {u:i for i,u in enumerate(users)}
    mid2col = {m:j for j,m in enumerate(items)}

    rows = df["user_id"].map(uid2row).to_numpy()
    cols = df["movie_id"].map(mid2col).to_numpy()
    data = np.ones(len(df), dtype=np.float32)
    R = sparse.csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))

    # cosine similarity via co-occurrence
    C = (R.T @ R).astype(np.float32)
    norms = np.sqrt(C.diagonal()).astype(np.float32)
    denom = norms[:,None]*norms[None,:]; denom[denom==0]=1.0
    S = C/denom
    S.setdiag(0.0)

    if MIN_OVERLAP > 1:
        mask = C.A < MIN_OVERLAP
        S = S.tolil()
        S[mask] = 0.0
        S = S.tocsr()

    pop = build_popularity(items)
    return items, mid2col, S.tocsr(), pop, R

def recommend_knn(S, items, mid2col, liked_ids, pop=None, k=10):
    if not items or S.shape[0]==0:
        return []
    cols = [mid2col[m] for m in liked_ids if m in mid2col]
    if not cols:
        # fall back to popularity
        order = np.argsort(-(pop if pop is not None else np.zeros(len(items))))
        return [items[i] for i in order[:k]]

    scores = np.asarray(S[:, cols].sum(axis=1)).ravel()
    # exclude liked
    for m in liked_ids:
        j = mid2col.get(m); 
        if j is not None: scores[j] = -np.inf

    if pop is not None and BLEND_POP > 0:
        sc = scores.copy(); sc[np.isneginf(sc)]=0.0
        if sc.max()>0: sc = sc/sc.max()
        final = (1-BLEND_POP)*sc + BLEND_POP*pop
    else:
        final = scores

    order = np.argsort(-final)
    out = []
    for idx in order:
        if len(out)>=k: break
        if np.isneginf(scores[idx]): continue
        out.append(items[idx])
    return out

def main():
    # pull & sort by id as time proxy
    df_all = pd.DataFrame(list(Myrating.objects.values("id","user_id","movie_id","rating"))).sort_values("id")
    df_all = df_all[df_all["rating"] > 0]

    if df_all.empty:
        print("No ratings yet.")
        return

    # per-user 80/20 split
    train_parts, test_parts = [], []
    for uid, grp in df_all.groupby("user_id"):
        n = len(grp); cut = max(int(n*0.8), 1)
        train_parts.append(grp.iloc[:cut])
        if cut < n: test_parts.append(grp.iloc[cut:])
    train = pd.concat(train_parts, ignore_index=True)
    test  = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=df_all.columns)

    # keep users with >=3 train and >=1 test
    train_ct = train.groupby("user_id").size()
    test_ct  = test.groupby("user_id").size()
    eval_users = sorted(set(train_ct[train_ct>=3].index) & set(test_ct[test_ct>=1].index))
    if not eval_users:
        print("Need users with >=3 train and >=1 test ratings.")
        return

    items, mid2col, S, pop, _ = build_knn(train)

    # relevance per user from test
    relevant = {uid: list(test.loc[test["user_id"]==uid, "movie_id"].values) for uid in eval_users}
    # liked in train per user
    liked_in_train = {
        uid: list(train[(train["user_id"]==uid) & (train["rating"]>=LIKE_THRESHOLD)]["movie_id"].values)
        for uid in eval_users
    }
    # exclude set per user
    seen_in_train = {
        uid: set(train.loc[train["user_id"]==uid, "movie_id"].values)
        for uid in eval_users
    }

    rows = []
    for uid in eval_users:
        rel = relevant[uid]
        liked = liked_in_train[uid]
        if not liked:
            # fall back to popularity filtered
            base_order = np.argsort(-pop) if pop is not None else np.arange(len(items))
            rec = [items[i] for i in base_order if items[i] not in seen_in_train[uid]][:K]
        else:
            rec = recommend_knn(S, items, mid2col, liked, pop, K)

        p, r = precision_recall_at_k(rec, rel, K)
        n = ndcg_at_k(rec, rel, K)
        rows.append({"user_id": uid, "prec": p, "rec": r, "ndcg": n})

    res = pd.DataFrame(rows)
    print("\n=== Cosine-KNN @K=10 (bin likes>=4, min_overlap=%d, blend_pop=%.2f) ===" % (MIN_OVERLAP, BLEND_POP))
    print(res[["prec","rec","ndcg"]].mean().to_frame("mean").round(4))

if __name__ == "__main__":
    main()
