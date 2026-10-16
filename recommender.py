"""Content-Based (TF-IDF) + Collaborative (SVD) + Hybrid recommender."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

try:  # optional: scikit-surprise
    from surprise import SVD, Dataset, Reader
    HAS_SURPRISE = True
except Exception:
    HAS_SURPRISE = False


def _tok(s):
    return str(s).replace(" ", "").replace(".", "").lower()


def _soup(row):
    genres = [_tok(g) for g in row.genres.split("|") if g and "no genres" not in g]
    cast = [_tok(c) for c in str(row.cast).split(",") if c.strip()]
    kws = [_tok(k) for k in str(row.keywords).split(",") if k.strip()]
    return " ".join(genres * 2 + cast + kws)  # genres ko 2x weight


def _norm(x):
    x = np.asarray(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min() + 1e-9)


# ------------------------------------------------------------------ Content-Based
class ContentBased:
    def __init__(self, movies):
        self.movies = movies.reset_index(drop=True)
        self.idx = {m: i for i, m in enumerate(self.movies.movieId)}
        soups = self.movies.apply(_soup, axis=1)
        self.tokens = [set(s.split()) for s in soups]
        self.X = TfidfVectorizer(token_pattern=r"[^ ]+").fit_transform(soups)  # L2-normalised rows

    def similar(self, movie_id):
        """Cosine similarity of one movie to every movie."""
        i = self.idx[movie_id]
        s = (self.X @ self.X[i].T).toarray().ravel()
        s[i] = -1
        return s

    def user_scores(self, history):
        """User profile = TF-IDF rows weighted by (rating - user mean); score = cosine(profile, movie)."""
        h = history[history.index.isin(self.idx)]
        if h.empty:
            return np.zeros(len(self.movies))
        w = (h - h.mean()).values
        if np.allclose(w, 0):
            w = h.values
        profile = self.X[[self.idx[m] for m in h.index]].T.dot(w)
        n = np.linalg.norm(profile)
        return np.asarray(self.X.dot(profile)).ravel() / n if n else np.zeros(len(self.movies))


# ------------------------------------------------------------------ Collaborative
class CollaborativeFiltering:
    """SVD matrix factorisation. Surprise agar installed ho to wahi, warna sklearn TruncatedSVD."""

    def __init__(self, n_factors=30):
        self.k = n_factors

    def fit(self, ratings, movie_ids):
        self.movie_ids = np.asarray(movie_ids)
        self.mid = {m: i for i, m in enumerate(self.movie_ids)}
        self.backend = "surprise" if HAS_SURPRISE else "sklearn"
        if HAS_SURPRISE:
            data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], Reader(rating_scale=(0.5, 5)))
            self.model = SVD(n_factors=self.k, random_state=42).fit(data.build_full_trainset())
            return self
        users = ratings.userId.unique()
        self.uid = {u: i for i, u in enumerate(users)}
        r = ratings[ratings.movieId.isin(self.mid)]
        self.global_mean = r.rating.mean()
        self.user_mean = r.groupby("userId").rating.mean()
        self.item_mean = r.groupby("movieId").rating.mean().reindex(self.movie_ids).fillna(self.global_mean).values
        centered = r.rating - r.userId.map(self.user_mean)
        mat = csr_matrix((centered, (r.userId.map(self.uid), r.movieId.map(self.mid))),
                         shape=(len(users), len(self.movie_ids)))
        svd = TruncatedSVD(n_components=min(self.k, len(self.movie_ids) - 1), random_state=42)
        self.U = svd.fit_transform(mat)
        self.Vt = svd.components_
        return self

    def predict_all(self, user_id):
        """Predicted rating (1-5) for every movie."""
        if self.backend == "surprise":
            return np.array([self.model.predict(user_id, m).est for m in self.movie_ids])
        if user_id not in self.uid:
            return self.item_mean.copy()
        u = self.uid[user_id]
        return np.clip(self.user_mean[user_id] + self.U[u] @ self.Vt, 0.5, 5)


# ------------------------------------------------------------------ Hybrid
class Hybrid:
    def __init__(self, movies, ratings, n_factors=30):
        self.movies = movies.reset_index(drop=True)
        self.ratings = ratings
        self.cb = ContentBased(self.movies)
        self.cf = CollaborativeFiltering(n_factors).fit(ratings, self.movies.movieId.values)
        self.by_user = {u: g.set_index("movieId").rating for u, g in ratings.groupby("userId")}
        stats = ratings.groupby("movieId").rating.agg(["mean", "count"]).reindex(self.movies.movieId)
        m = stats["mean"].mean()
        self.popularity = ((stats["count"].fillna(0) * stats["mean"].fillna(m) + 5 * m) / (stats["count"].fillna(0) + 5)).values

    def history(self, user_id):
        return self.by_user.get(user_id, pd.Series(dtype=float))

    def scores(self, user_id, mode="hybrid", alpha=0.6, adaptive=True):
        """mode: content | collab | hybrid. alpha = weight of collaborative part."""
        h = self.history(user_id)
        if h.empty:  # cold start -> popularity
            return self.popularity, np.zeros(len(self.movies)), self.popularity
        cb, cf = self.cb.user_scores(h), self.cf.predict_all(user_id)
        if mode == "content":
            s = cb
        elif mode == "collab":
            s = cf
        else:
            a = alpha * min(1, len(h) / 15) if adaptive else alpha  # kam ratings -> content pe zyada bharosa
            s = a * _norm(cf) + (1 - a) * _norm(cb)
        return s, cb, cf

    def recommend(self, user_id, mode="hybrid", alpha=0.6, adaptive=True, n=10, genre=None):
        h = self.history(user_id)
        s, cb, cf = self.scores(user_id, mode, alpha, adaptive)
        s = np.where(self.movies.movieId.isin(h.index), -np.inf, s)
        if genre:
            s = np.where(self.movies.genres.str.contains(genre, regex=False), s, -np.inf)
        top = [i for i in np.argsort(-s)[:n] if np.isfinite(s[i])]
        liked = h[h >= 4] if (h >= 4).any() else h.nlargest(3)
        liked_rows = [self.cb.idx[m] for m in liked.index if m in self.cb.idx]
        out = []
        for i in top:
            reason = "Popular & highly rated (cold start)"
            if liked_rows:
                sims = (self.cb.X[i] @ self.cb.X[liked_rows].T).toarray().ravel()
                j = liked_rows[int(sims.argmax())]
                shared = sorted(self.cb.tokens[i] & self.cb.tokens[j])[:4]
                reason = f"Kyunki aapko '{self.movies.title[j]}' pasand aayi" + (f" (common: {', '.join(shared)})" if shared else "")
            out.append((self.movies.title[i], self.movies.genres[i].replace("|", ", "),
                        round(float(s[i]), 3), round(float(cb[i]), 3), round(float(cf[i]), 2), reason))
        return pd.DataFrame(out, columns=["title", "genres", "score", "content_score", "cf_pred_rating", "reason"])

    def similar_movies(self, movie_id, n=10):
        s = self.cb.similar(movie_id)
        top = np.argsort(-s)[:n]
        return pd.DataFrame({"title": self.movies.title[top].values,
                             "genres": self.movies.genres[top].str.replace("|", ", ").values,
                             "similarity": np.round(s[top], 3)})


# ------------------------------------------------------------------ Evaluation
def evaluate(movies, ratings, k=10, alpha=0.6, test_frac=0.2, max_users=100, seed=42):
    """Per-user hold-out. RMSE (CF) + Precision@k / Recall@k (rating>=4 = relevant) for 3 modes."""
    test = ratings.groupby("userId").sample(frac=test_frac, random_state=seed)
    train = ratings.drop(test.index)
    model = Hybrid(movies, train)
    users = test.userId.drop_duplicates().sample(min(max_users, test.userId.nunique()), random_state=seed)
    mids = model.cf.mid
    sq, res = [], {m: [] for m in ("content", "collab", "hybrid")}
    for u in users:
        t = test[test.userId == u]
        preds = model.cf.predict_all(u)
        sq += [(preds[mids[m]] - r) ** 2 for m, r in zip(t.movieId, t.rating) if m in mids]
        rel = set(t[t.rating >= 4].movieId)
        if not rel:
            continue
        seen = model.history(u).index
        for mode in res:
            s = np.where(model.movies.movieId.isin(seen), -np.inf, model.scores(u, mode, alpha)[0])
            top = set(model.movies.movieId.values[np.argsort(-s)[:k]])
            hits = len(top & rel)
            res[mode].append((hits / k, hits / len(rel)))
    df = pd.DataFrame({m: np.mean(v, axis=0) for m, v in res.items()}, index=[f"Precision@{k}", f"Recall@{k}"]).T
    df["RMSE (CF)"] = [np.nan, float(np.sqrt(np.mean(sq))), np.nan]
    return df.round(3)
