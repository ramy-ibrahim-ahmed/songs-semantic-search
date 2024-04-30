import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from preprocessing import preprocess_text

# import torch


class CosineSimilarity:
    """
    1- Calculate the cosine similarity range from (-1, 1).
    2- Specify dims for norms across all elements in the vector.
    3- Handeling division by 0 by adding small epsilon.
    """
    def __init__(self, dim=1, eps=1e-8):
        self.dim = dim
        self.eps = eps

    def __call__(self, x1, x2):
        x1_norm = np.linalg.norm(x1, axis=self.dim, keepdims=True) + self.eps
        x2_norm = np.linalg.norm(x2, axis=self.dim, keepdims=True) + self.eps
        x1 = x1 / x1_norm
        x2 = x2 / x2_norm

        cos_sim = np.sum(x1 * x2, axis=self.dim)
        return cos_sim


data = pd.read_csv(r"data.csv")
data["lyrics_processed"] = data["Lyric"].apply(preprocess_text)

MODEL = SentenceTransformer("all-mpnet-base-v2")  # all-MiniLM-L6-v2
# embedding_matrix = MODEL.encode(data["lyrics_processed"].tolist(), convert_to_tensor=True)
# torch.save(embedding_matrix, "embedding_matrix.pt")