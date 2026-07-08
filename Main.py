"""
Two-Tower Movie Recommender — inference API.

Loads the trained model + lookup artifacts ONCE at startup (not per request),
keeps everything in memory, and serves fast recommendations over HTTP.

Expected files in ARTIFACTS_DIR (default: "artifacts/"), all produced by the
notebook (see the "Save Artifacts For Deployment" cell you add before export):
  - model_state_dict.pth
  - model_config.json
  - user_to_index.pkl
  - movie_to_index.pkl
  - movie_genre_map.pkl
  - movies_meta.csv         (movieId, title, genres)
  - popular_movie_ids.pkl   (list[int], most-rated movies first, for cold start)

Run locally:
    uvicorn Main:app --reload --port 8000

Endpoints:
    GET /health
    GET /recommend?user_id=123&top_k=10
    GET /similar?title=Toy%20Story&top_k=10
    GET /movies/search?q=toy&limit=8
"""

import json
import os
import pickle
from typing import List, Optional

import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")
device = torch.device("cpu")  # inference-only, CPU is plenty for this model size


# ---------------------------------------------------------------------------
# Model definition — identical architecture to the notebook, so state_dict
# loads in cleanly. Kept here directly (rather than saving the whole pickled
# object) so the API never depends on unpickling a class from another module.
# ---------------------------------------------------------------------------
class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_movies, num_genres, embed_dim=32, hidden_dim=32, output_dim=32):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.user_layers = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
        )
        self.movie_embedding = nn.Embedding(num_movies, embed_dim)
        self.genre_embedding = nn.Embedding(num_genres, embed_dim, padding_idx=0)
        self.movie_layers = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
        )
        self.output_layer = nn.Linear(1, 1)

    def forward_user(self, user_idx):
        return self.user_layers(self.user_embedding(user_idx))

    def forward_movie(self, movie_idx, genre_idx):
        m_emb = self.movie_embedding(movie_idx)
        g_emb = self.genre_embedding(genre_idx).mean(dim=1)
        return self.movie_layers(torch.cat([m_emb, g_emb], dim=1))

    def forward(self, user_idx, movie_idx, genre_idx):
        user_vec = self.forward_user(user_idx)
        movie_vec = self.forward_movie(movie_idx, genre_idx)
        interaction = (user_vec * movie_vec).sum(dim=1, keepdim=True)
        return self.output_layer(interaction).squeeze()


app = FastAPI(title="Two-Tower Movie Recommender API")

# Wide-open CORS for a demo. Tighten this to your Vercel domain before sharing
# the API URL widely (see README).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_state = {}


@app.on_event("startup")
def load_artifacts():
    def path(name):
        return os.path.join(ARTIFACTS_DIR, name)

    with open(path("user_to_index.pkl"), "rb") as f:
        user_to_index = pickle.load(f)
    with open(path("movie_to_index.pkl"), "rb") as f:
        movie_to_index = pickle.load(f)
    with open(path("movie_genre_map.pkl"), "rb") as f:
        movie_genre_map = pickle.load(f)
    with open(path("model_config.json")) as f:
        model_cfg = json.load(f)
    with open(path("popular_movie_ids.pkl"), "rb") as f:
        popular_movie_ids = pickle.load(f)

    model = TwoTowerModel(
        num_users=model_cfg["num_users"],
        num_movies=model_cfg["num_movies"],
        num_genres=model_cfg["num_genres"],
        embed_dim=model_cfg["embed_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        output_dim=model_cfg["output_dim"],
    )
    state_dict = torch.load(path("model_state_dict.pth"), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    movies_df = pd.read_csv(path("movies_meta.csv"))
    movie_id_to_title = dict(zip(movies_df.movieId, movies_df.title))
    movie_id_to_genres = dict(zip(movies_df.movieId, movies_df.genres))

    # Precompute every movie's tower vector ONCE. Every /recommend and
    # /similar call afterwards just does a dot product / cosine sim against
    # this cached matrix — no repeated forward passes through movie_layers.
    all_movie_indices = list(movie_to_index.values())
    all_genre_indices = [movie_genre_map[i] for i in all_movie_indices]
    movie_tensor = torch.tensor(all_movie_indices)
    genre_tensor = torch.tensor(all_genre_indices)
    with torch.no_grad():
        all_movie_vecs = model.forward_movie(movie_tensor, genre_tensor)

    index_to_movie = {v: k for k, v in movie_to_index.items()}
    movieidx_to_row = {m: i for i, m in enumerate(all_movie_indices)}

    _state.update(
        model=model,
        user_to_index=user_to_index,
        movie_to_index=movie_to_index,
        movie_id_to_title=movie_id_to_title,
        movie_id_to_genres=movie_id_to_genres,
        popular_movie_ids=popular_movie_ids,
        all_movie_indices=all_movie_indices,
        all_movie_vecs=all_movie_vecs,
        index_to_movie=index_to_movie,
        movieidx_to_row=movieidx_to_row,
        movies_df=movies_df,
    )
    print(f"Loaded model + artifacts. users={len(user_to_index):,} movies={len(movie_to_index):,}")


class MovieOut(BaseModel):
    movieId: int
    title: str
    genres: Optional[str] = None
    score: Optional[float] = None


class NewUserRequest(BaseModel):
    liked_titles: List[str]
    top_k: int = 10


@app.get("/health")
def health():
    return {
        "status": "ok",
        "users": len(_state.get("user_to_index", {})),
        "movies": len(_state.get("movie_to_index", {})),
    }


@app.get("/recommend", response_model=List[MovieOut])
def recommend(user_id: int = Query(...), top_k: int = Query(10, ge=1, le=50)):
    user_to_index = _state["user_to_index"]
    movie_id_to_title = _state["movie_id_to_title"]
    movie_id_to_genres = _state["movie_id_to_genres"]

    if user_id not in user_to_index:
        # Cold start: no personalized score, fall back to most-rated movies.
        ids = _state["popular_movie_ids"][:top_k]
        return [
            MovieOut(
                movieId=mid,
                title=movie_id_to_title.get(mid, str(mid)),
                genres=movie_id_to_genres.get(mid),
                score=None,
            )
            for mid in ids
        ]

    model = _state["model"]
    u_idx = user_to_index[user_id]
    with torch.no_grad():
        user_vec = model.forward_user(torch.tensor([u_idx]))
        scores = (user_vec * _state["all_movie_vecs"]).sum(dim=1)

    top_scores, top_idx = torch.topk(scores, min(top_k, len(scores)))
    index_to_movie = _state["index_to_movie"]

    out = []
    for score, idx in zip(top_scores, top_idx):
        mid = index_to_movie[idx.item()]
        out.append(
            MovieOut(
                movieId=mid,
                title=movie_id_to_title.get(mid, str(mid)),
                genres=movie_id_to_genres.get(mid),
                score=round(score.item(), 4),
            )
        )
    return out


@app.get("/similar", response_model=List[MovieOut])
def similar(title: str = Query(..., min_length=1), top_k: int = Query(10, ge=1, le=50)):
    movies_df = _state["movies_df"]
    movie_to_index = _state["movie_to_index"]
    movie_id_to_title = _state["movie_id_to_title"]
    movie_id_to_genres = _state["movie_id_to_genres"]

    matches = movies_df[movies_df["title"].str.contains(title, case=False, na=False, regex=False)]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"No movie found matching '{title}'")

    match_row = matches.iloc[0]
    query_movie_id = int(match_row["movieId"])
    query_idx = movie_to_index[query_movie_id]

    all_movie_vecs = _state["all_movie_vecs"]
    movieidx_to_row = _state["movieidx_to_row"]
    query_vec = all_movie_vecs[movieidx_to_row[query_idx]].unsqueeze(0)

    sims = torch.nn.functional.cosine_similarity(query_vec, all_movie_vecs)
    top_scores, top_pos = torch.topk(sims, min(top_k + 1, len(sims)))

    all_movie_indices = _state["all_movie_indices"]
    index_to_movie = _state["index_to_movie"]

    results = []
    for score, pos in zip(top_scores, top_pos):
        idx = all_movie_indices[pos.item()]
        mid = index_to_movie[idx]
        if mid == query_movie_id:
            continue
        results.append(
            MovieOut(
                movieId=mid,
                title=movie_id_to_title.get(mid, str(mid)),
                genres=movie_id_to_genres.get(mid),
                score=round(score.item(), 4),
            )
        )
        if len(results) == top_k:
            break
    return results


@app.post("/recommend/new-user", response_model=List[MovieOut])
def recommend_new_user(req: NewUserRequest):
    """
    Cold-start recommendations for a brand-new user who has no trained
    user_embedding row yet. Instead of a learned user vector, this averages
    the (already-cached) movie-tower vectors of a few movies they say they
    like, and scores every movie against that 'pseudo user vector'.
    """
    movies_df = _state["movies_df"]
    movie_to_index = _state["movie_to_index"]
    movieidx_to_row = _state["movieidx_to_row"]
    all_movie_vecs = _state["all_movie_vecs"]
    all_movie_indices = _state["all_movie_indices"]
    index_to_movie = _state["index_to_movie"]
    movie_id_to_title = _state["movie_id_to_title"]
    movie_id_to_genres = _state["movie_id_to_genres"]

    if not req.liked_titles:
        raise HTTPException(status_code=400, detail="Provide at least one liked_titles entry.")

    liked_ids = []
    for t in req.liked_titles:
        matches = movies_df[movies_df["title"].str.contains(t, case=False, na=False, regex=False)]
        if not matches.empty:
            liked_ids.append(int(matches.iloc[0]["movieId"]))

    if not liked_ids:
        # None of the typed titles matched anything in the catalog — fall
        # back to popularity, same as an unknown user_id in /recommend.
        ids = _state["popular_movie_ids"][: req.top_k]
        return [
            MovieOut(
                movieId=mid,
                title=movie_id_to_title.get(mid, str(mid)),
                genres=movie_id_to_genres.get(mid),
                score=None,
            )
            for mid in ids
        ]

    rows = [movieidx_to_row[movie_to_index[mid]] for mid in liked_ids]
    with torch.no_grad():
        pseudo_user_vec = all_movie_vecs[rows].mean(dim=0, keepdim=True)
        scores = (pseudo_user_vec * all_movie_vecs).sum(dim=1)

    liked_set = set(liked_ids)
    top_scores, top_pos = torch.topk(scores, min(req.top_k + len(liked_ids), len(scores)))

    results = []
    for score, pos in zip(top_scores, top_pos):
        idx = all_movie_indices[pos.item()]
        mid = index_to_movie[idx]
        if mid in liked_set:
            continue
        results.append(
            MovieOut(
                movieId=mid,
                title=movie_id_to_title.get(mid, str(mid)),
                genres=movie_id_to_genres.get(mid),
                score=round(score.item(), 4),
            )
        )
        if len(results) == req.top_k:
            break
    return results


@app.get("/movies/search", response_model=List[MovieOut])
def search_movies(q: str = Query(..., min_length=1), limit: int = Query(8, ge=1, le=25)):
    """Powers a title-autocomplete box in the frontend."""
    movies_df = _state["movies_df"]
    matches = movies_df[movies_df["title"].str.contains(q, case=False, na=False, regex=False)].head(limit)
    return [
        MovieOut(movieId=int(r.movieId), title=r.title, genres=r.genres) for r in matches.itertuples()
    ]
