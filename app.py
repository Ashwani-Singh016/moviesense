import pandas as pd
import streamlit as st

from data_utils import load_data
from recommender import HAS_SURPRISE, Hybrid, evaluate

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
NEW_USER = -1


@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_model(extra=()):
    movies, ratings, _ = get_data()
    if extra:  # new user ki ratings add karke model refit
        add = pd.DataFrame([(NEW_USER, m, r) for m, r in extra], columns=["userId", "movieId", "rating"])
        ratings = pd.concat([ratings, add], ignore_index=True)
    return Hybrid(movies, ratings)


movies, ratings, source = get_data()
title2id = dict(zip(movies.title, movies.movieId))

st.title("🎬 Smart Movie Recommender")
st.caption(f"Data: **{source}** · {len(movies)} movies · {len(ratings)} ratings · "
           f"CF backend: {'Surprise SVD' if HAS_SURPRISE else 'scikit-learn TruncatedSVD'}")

# ---------------- Sidebar
with st.sidebar:
    st.header("Settings")
    mode_label = st.radio("Model", ["Hybrid", "Content-Based", "Collaborative"])
    mode = {"Hybrid": "hybrid", "Content-Based": "content", "Collaborative": "collab"}[mode_label]
    alpha = st.slider("Hybrid: Collaborative weight (α)", 0.0, 1.0, 0.6, 0.05, disabled=mode != "hybrid")
    adaptive = st.checkbox("Adaptive α (kam ratings → content zyada)", True, disabled=mode != "hybrid")
    n = st.slider("Kitni recommendations", 5, 20, 10)
    genre = st.selectbox("Genre filter", ["All"] + sorted({g for gs in movies.genres for g in gs.split("|")}))
    genre = None if genre == "All" else genre

# ---------------- User selection
who = st.radio("Aap kaun?", ["Existing user", "New user (khud rate karein)"], horizontal=True)
if who == "Existing user":
    user_id = st.selectbox("User ID", sorted(ratings.userId.unique())[:500])
    model = get_model()
else:
    picked = st.multiselect("Wo movies chuno jo aap dekh chuke ho", sorted(title2id))
    extra = tuple((title2id[t], st.slider(t, 1.0, 5.0, 4.0, 0.5, key=t)) for t in picked)
    user_id, model = NEW_USER, get_model(extra)

tab1, tab2, tab3, tab4 = st.tabs(["⭐ Recommendations", "🔍 Similar movies", "📜 History", "📊 Evaluation"])

with tab1:
    recs = model.recommend(user_id, mode, alpha, adaptive, n, genre)
    if model.history(user_id).empty:
        st.info("Abhi koi rating nahi hai - popular movies dikh rahi hain. Kuch movies rate karo!")
    st.dataframe(recs, width="stretch", hide_index=True)
    if not recs.empty:
        st.bar_chart(recs.set_index("title")["score"])

with tab2:
    pick = st.selectbox("Movie chuno", sorted(title2id))
    st.dataframe(model.similar_movies(title2id[pick], n), width="stretch", hide_index=True)

with tab3:
    h = model.history(user_id)
    hist = movies.set_index("movieId").loc[h.index, ["title", "genres"]].assign(rating=h.values)
    st.dataframe(hist.sort_values("rating", ascending=False), width="stretch", hide_index=True)

with tab4:
    st.write("20% ratings hold-out karke teeno models ki Precision@K / Recall@K (rating ≥ 4 = relevant) aur CF ka RMSE.")
    k = st.slider("K", 5, 20, 10, key="k")
    if st.button("Run evaluation"):
        with st.spinner("Evaluating..."):
            st.dataframe(evaluate(movies, ratings, k=k, alpha=alpha))
