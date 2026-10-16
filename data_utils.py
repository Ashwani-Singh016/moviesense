"""Data loading. MovieLens (data/movies.csv + data/ratings.csv [+ tags.csv]) use hota hai agar mile,
warna built-in sample dataset (36 movies, 120 synthetic users) chalta hai."""
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# title|genres|cast|keywords
SAMPLE = """Inception (2010)|Action,Sci-Fi,Thriller|Leonardo DiCaprio,Joseph Gordon-Levitt,Tom Hardy|heist,dreams,mind-bending
Interstellar (2014)|Adventure,Drama,Sci-Fi|Matthew McConaughey,Anne Hathaway,Michael Caine|space,time,family
The Dark Knight (2008)|Action,Crime,Drama|Christian Bale,Heath Ledger,Michael Caine|superhero,joker,gotham
Parasite (2019)|Comedy,Drama,Thriller|Song Kang-ho,Choi Woo-shik|class,twist,korean
Spirited Away (2001)|Animation,Fantasy|Rumi Hiiragi,Miyu Irino|anime,spirits,coming-of-age
The Grand Budapest Hotel (2014)|Comedy,Crime|Ralph Fiennes,Tony Revolori|quirky,heist,europe
Mad Max: Fury Road (2015)|Action,Adventure,Sci-Fi|Tom Hardy,Charlize Theron|dystopia,chase,desert
Whiplash (2014)|Drama,Music|Miles Teller,J.K. Simmons|obsession,jazz,mentor
La La Land (2016)|Comedy,Drama,Romance|Ryan Gosling,Emma Stone|jazz,dreams,la
The Shawshank Redemption (1994)|Crime,Drama|Tim Robbins,Morgan Freeman|prison,hope,friendship
Pulp Fiction (1994)|Crime,Thriller|John Travolta,Samuel L. Jackson,Uma Thurman|nonlinear,hitmen,dialogue
Everything Everywhere All at Once (2022)|Action,Comedy,Sci-Fi|Michelle Yeoh,Ke Huy Quan|multiverse,family,absurd
Knives Out (2019)|Comedy,Crime,Mystery|Daniel Craig,Ana de Armas,Chris Evans|whodunit,family,twist
Get Out (2017)|Horror,Mystery,Thriller|Daniel Kaluuya,Allison Williams|race,twist,social
The Social Network (2010)|Biography,Drama|Jesse Eisenberg,Andrew Garfield|startup,tech,betrayal
Toy Story (1995)|Animation,Children,Comedy|Tom Hanks,Tim Allen|toys,friendship,family
Your Name (2016)|Animation,Romance|Ryunosuke Kamiki,Mone Kamishiraishi|anime,time,fate
Blade Runner 2049 (2017)|Drama,Sci-Fi|Ryan Gosling,Harrison Ford|dystopia,androids,neon
John Wick (2014)|Action,Crime,Thriller|Keanu Reeves,Willem Dafoe|hitmen,revenge,stylish
The Princess Bride (1987)|Adventure,Comedy,Fantasy|Cary Elwes,Robin Wright|adventure,romance,quotable
Gone Girl (2014)|Drama,Mystery,Thriller|Ben Affleck,Rosamund Pike|marriage,twist,media
Dune (2021)|Adventure,Sci-Fi|Timothee Chalamet,Zendaya|desert,epic,politics
3 Idiots (2009)|Comedy,Drama|Aamir Khan,R. Madhavan,Sharman Joshi|india,college,friendship
Dangal (2016)|Biography,Drama,Sports|Aamir Khan,Fatima Sana Shaikh|india,wrestling,family
Lagaan (2001)|Drama,Musical,Sports|Aamir Khan,Gracy Singh|india,cricket,underdog
Gangs of Wasseypur (2012)|Action,Crime,Drama|Manoj Bajpayee,Nawazuddin Siddiqui|india,revenge,epic
Zindagi Na Milegi Dobara (2011)|Comedy,Drama|Hrithik Roshan,Farhan Akhtar,Abhay Deol|india,travel,friendship
Andhadhun (2018)|Comedy,Crime,Thriller|Ayushmann Khurrana,Tabu|india,twist,music
Gladiator (2000)|Action,Adventure,Drama|Russell Crowe,Joaquin Phoenix|rome,revenge,epic
Avengers: Endgame (2019)|Action,Adventure,Sci-Fi|Robert Downey Jr.,Chris Evans|superhero,time,epic
The Notebook (2004)|Drama,Romance|Ryan Gosling,Rachel McAdams|love,memory,tearjerker
Titanic (1997)|Drama,Romance|Leonardo DiCaprio,Kate Winslet|ship,love,disaster
The Conjuring (2013)|Horror,Thriller|Vera Farmiga,Patrick Wilson|haunted,paranormal,true-story
Superbad (2007)|Comedy|Jonah Hill,Michael Cera|teen,party,friendship
Coco (2017)|Animation,Children,Fantasy|Anthony Gonzalez,Gael Garcia Bernal|music,family,afterlife
Se7en (1995)|Crime,Mystery,Thriller|Brad Pitt,Morgan Freeman|serial-killer,detective,dark"""


def _sample():
    rows = [l.split("|") for l in SAMPLE.split("\n")]
    movies = pd.DataFrame({
        "movieId": range(1, len(rows) + 1),
        "title": [r[0] for r in rows],
        "genres": [r[1].replace(",", "|") for r in rows],
        "cast": [r[2] for r in rows],
        "keywords": [r[3] for r in rows],
    })
    rng = np.random.default_rng(42)
    all_g = sorted({g for gs in movies.genres for g in gs.split("|")})
    recs = []
    for u in range(1, 121):
        fav = set(rng.choice(all_g, size=3, replace=False))
        for m in rng.choice(movies.movieId, size=rng.integers(12, 24), replace=False):
            overlap = len(fav & set(movies.genres[m - 1].split("|")))
            base = 2.2 if overlap == 0 else 3.0 + 0.8 * min(overlap, 2)
            r = np.clip(round((base + rng.normal(0, 0.5)) * 2) / 2, 1, 5)
            recs.append((u, m, r))
    return movies, pd.DataFrame(recs, columns=["userId", "movieId", "rating"]), "Sample (synthetic)"


def load_data():
    """Returns (movies, ratings, source_name). movies: movieId,title,genres('|'),cast,keywords"""
    mp, rp, tp = (os.path.join(DATA_DIR, f) for f in ("movies.csv", "ratings.csv", "tags.csv"))
    if not (os.path.exists(mp) and os.path.exists(rp)):
        return _sample()
    movies = pd.read_csv(mp)
    ratings = pd.read_csv(rp)[["userId", "movieId", "rating"]]
    for c in ("cast", "keywords"):  # MovieLens me cast nahi hota; optional columns
        if c not in movies:
            movies[c] = ""
    if os.path.exists(tp) and not movies["keywords"].astype(bool).any():
        tags = pd.read_csv(tp)
        tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()
        kw = tags.groupby("movieId")["tag"].apply(lambda s: ",".join(s.value_counts().index[:15]))
        movies["keywords"] = movies["movieId"].map(kw).fillna("")
    return movies.fillna(""), ratings, "MovieLens"
