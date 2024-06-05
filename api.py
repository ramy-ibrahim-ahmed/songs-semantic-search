from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import torch
from main import MODEL
from main import CosineSimilarity

api = FastAPI()

# Load data and model at startup
data = pd.read_csv("data.csv")
embedding_matrix = torch.load("embedding_matrix.pt")


# Data model for incoming requests
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


# Utility function to get similar songs
def get_similar_songs(query: str, k: int):
    user_query_vector = MODEL.encode([query], convert_to_tensor=True)
    cos = CosineSimilarity()
    similarity_scores = cos(np.array(user_query_vector), np.array(embedding_matrix))
    return similarity_scores


def format_similar_songs(data, similarity_scores, k):
    top_similar_song_indices = np.argpartition(similarity_scores, -k)[-k:]
    top_similar_song_indices = top_similar_song_indices[
        np.argsort(similarity_scores[top_similar_song_indices])
    ][::-1]

    top_k_songs = []
    for index in top_similar_song_indices:
        SONG = data.loc[int(index), "Title"]
        ARTIST = data.loc[int(index), "Artist"]
        ALBUM = data.loc[int(index), "Album"]
        score = similarity_scores[index].item()
        top_k_songs.append(
            {"song": SONG, "artist": ARTIST, "album": ALBUM, "score": score}
        )

    return top_k_songs


@api.get("/")
def read_root():
    return {"message": "Welcome to the Song Search API"}


@api.post("/search")
def search_song(query_request: QueryRequest):
    query = query_request.query
    top_k = query_request.top_k

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    similarity_scores = get_similar_songs(query, top_k)
    most_similar_song_index = np.argmax(similarity_scores)

    SONG = data.loc[int(most_similar_song_index), "Title"]
    ARTIST = data.loc[int(most_similar_song_index), "Artist"]
    ALBUM = data.loc[int(most_similar_song_index), "Album"]
    DATE = data.loc[int(most_similar_song_index), "Date"]

    response = {
        "most_similar_song": {
            "song": SONG,
            "artist": ARTIST,
            "album": ALBUM,
            "date": DATE,
        },
        "top_k_similar_songs": format_similar_songs(data, similarity_scores, top_k),
    }

    return response