import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch

from sentence_transformers.util import cos_sim
from sklearn.decomposition import PCA
from main import MODEL
from preprocessing import preprocess_text

data = pd.read_csv("data.csv")
embeddings = MODEL.encode(data["Lyric"].head(5).apply(preprocess_text).tolist(), convert_to_tensor=True)

sim = np.zeros((5, 5))
for i in range(5):
    sim[i:, i] = cos_sim(embeddings[i], embeddings[i:])

# print(sim)
# sns.heatmap(sim, annot=True)
# plt.plot()
# plt.savefig("similarity.png")

pca = PCA(n_components=2)
projected_embeddings = pca.fit_transform(embeddings)

colors = ["red", "green", "blue", "purple", "orange"]
sns.scatterplot(
    x=projected_embeddings[:, 0],
    y=projected_embeddings[:, 1],
    s=50,
    alpha=0.7,
    color=colors,
)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("Sentence Embeddings (PCA)")
plt.grid(True, lw=0.5, linestyle="--")
# plt.savefig("2D.png")
# plt.show()

# embedding_matrix = torch.load("embedding_matrix.pt")
# projected_embeddings_all = pca.fit_transform(embedding_matrix)
# sns.scatterplot(
#     x=projected_embeddings_all[:, 0],
#     y=projected_embeddings_all[:, 1],
#     s=50,
#     alpha=0.7,
# )
# plt.xlabel("Dimension 1")
# plt.ylabel("Dimension 2")
# plt.title("Sentence Embeddings (PCA)")
# plt.grid(True, lw=0.5, linestyle="--")
# plt.savefig("2D_all_embeddings.png")
# plt.show()

print(MODEL)