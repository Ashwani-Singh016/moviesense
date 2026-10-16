# 🎬 Movie Recommendation System (Content + Collaborative + Hybrid)

**Stack:** Python, Pandas, Scikit-learn (TF-IDF, TruncatedSVD), optional Surprise (SVD), Streamlit

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Bina kisi data ke built-in sample dataset chalta hai.

## Real data (MovieLens)
1. https://grouplens.org/datasets/movielens/latest/ se `ml-latest-small` download karein.
2. `movies.csv`, `ratings.csv` (aur optional `tags.csv`) ko `data/` folder me rakhein.
3. App automatically MovieLens use karegi. (Tags -> plot keywords. Actors ke liye `movies.csv` me `cast` column (comma-separated) jod sakte hain, e.g. TMDB se.)

## Kaise kaam karta hai
- **Content-Based:** genres(2x) + cast + keywords -> TF-IDF. User profile = rated movies ka weighted average (rating - user mean). Score = cosine similarity.
- **Collaborative:** SVD matrix factorisation (Surprise ya sklearn TruncatedSVD) se user ka predicted rating.
- **Hybrid:** `α * norm(CF) + (1-α) * norm(Content)`. Adaptive mode me α = α * min(1, #ratings/15), yaani naye user ke liye content zyada, cold-start me popularity fallback.
- **Evaluation:** per-user hold-out, Precision@K, Recall@K, RMSE.

## Files
`app.py` (UI) · `recommender.py` (models + eval) · `data_utils.py` (data loading)
