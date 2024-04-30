import streamlit as st
import pandas as pd
import numpy as np
import torch
from main import MODEL
from main import CosineSimilarity

def display_similar_songs(data, similarity_scores, k):
    # Get top k similarities in form of indices by partitioning.
    top_similar_song_indices = np.argpartition(similarity_scores, -k)[-k:]
    # Sort indices based on the order that return accending similarities and retrive from -1.
    top_similar_song_indices = top_similar_song_indices[np.argsort(similarity_scores[top_similar_song_indices])][::-1]

    top_k_songs = []
    for index in top_similar_song_indices:
        SONG = data.loc[int(index), 'Title']
        ARTIST = data.loc[int(index), 'Artist']
        ALBUM = data.loc[int(index), 'Album']
        score = similarity_scores[index].item()
        top_k_songs.append((SONG, ARTIST, ALBUM, score))

    df = pd.DataFrame(top_k_songs, columns=["Song", "Artist", "Album", "Score"])
    st.subheader(f"Top {k} most relevant songs:")
    st.table(df)

def main():
    data = pd.read_csv("data.csv")
    embedding_matrix = torch.load("embedding_matrix.pt")

    st.title("Song Search")
    user_query = st.text_input("Enter your search:")

    if user_query:
        user_query_vector = MODEL.encode([user_query], convert_to_tensor=True)

        cos = CosineSimilarity()
        similarity_scores = cos(np.array(user_query_vector), np.array(embedding_matrix))

        most_similar_song_index = np.argmax(similarity_scores)
        SONG = data.loc[int(most_similar_song_index), "Title"]
        ARTIST = data.loc[int(most_similar_song_index), "Artist"]
        ALBUM = data.loc[int(most_similar_song_index), "Album"]
        DATE = data.loc[int(most_similar_song_index), "Date"]
        
        st.markdown(f'<p style="font-size: 24px; color: #008080;"><b>Are you looking for {SONG} by {ARTIST}?</b></p>', unsafe_allow_html=True)
        st.write(f"This song was released on {DATE}, Album : {ALBUM}")

        k = 5
        if st.button(f"Show top {k} similar songs"):
            display_similar_songs(data, similarity_scores, k)

if __name__ == "__main__":
    main()