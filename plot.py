import numpy as np
import pandas as pd
import seaborn as sb

# import matplotlib.pyplot as plt

from sentence_transformers.util import cos_sim
from main import MODEL
from preprocessing import preprocess_text

data = pd.read_csv("data.csv")
embeddings = MODEL.encode(data["Lyric"].head(5).apply(preprocess_text).tolist(), convert_to_tensor=True)

sim = np.zeros((5, 5))
for i in range(5):
    sim[i:, i] = cos_sim(embeddings[i], embeddings[i:])

# print(sim)
sb.heatmap(sim, annot=True)
# plt.plot()
# plt.savefig("similarity.png")
