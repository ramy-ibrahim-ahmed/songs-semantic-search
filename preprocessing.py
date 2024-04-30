import re
import nltk.corpus

# import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# file_paths = [
#     r"archive\csv\ArianaGrande.csv",
#     r"archive\csv\BillieEilish.csv",
#     r"archive\csv\Drake.csv",
#     r"archive\csv\Rihanna.csv",
#     r"archive\csv\SelenaGomez.csv",
#     r"archive\csv\TaylorSwift.csv",
#     r"archive\csv\DuaLipa.csv",
#     # r"archive\csv\Eminem.csv",
# ]
# dataframes = [pd.read_csv(path) for path in file_paths]
# merged_df = pd.concat(dataframes, axis=0, ignore_index=True)
# merged_df.to_csv('data.csv', index=False)


# nltk.download("punkt")
# nltk.download("stopwords")


def preprocess_text(text):
    """
    1- Lower case.
    2- Remove puncituations & special chars not in [^\w\s].
    3- Tokenize text to words.
    4- Remove Stop words in english that will not add to the meaning.
    5- Stemmer like making "running" back to its base word "run".
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    stop_words = nltk.corpus.stopwords.words("english")
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return stemmed_tokens


# df = pd.read_csv('data.csv')
# df['Title'] = df['Title'].str.replace('?', '')
# df.drop_duplicates(inplace=True)
# df = df[~df['Title'].str.contains('remix|party|live|(remix)|(party)|(live)', case=False)]
# df.to_csv('data.csv', index=False)