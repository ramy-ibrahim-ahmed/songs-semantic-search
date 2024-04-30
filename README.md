<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
  <h1>Song Lyrics Similarity Search</h1>

  <h2>Description</h2>

  <p>This project is a song lyrics similarity search application built using Streamlit. It allows users to enter a song query, and the app finds similar songs based on the cosine similarity of their lyrics embeddings.</p>

  <h2>Dependencies</h2>

  <ul>
    <li>pandas</li>
    <li>numpy</li>
    <li>streamlit</li>
    <li>sentence-transformers</li>
    <li>nltk</li>
  </ul>

  <h2>Installation</h2>

  <p>1. Install the required libraries using pip:</p>

  <pre>bash
  pip install pandas numpy streamlit sentence-transformers nltk
  </pre>

  <p>2. Download the pre-trained sentence transformer model (all-mpnet-base-v2) from <a href="https://huggingface.co/sentence-transformers/all-mpnet-base-v2">here</a>. Place the downloaded model files in a folder accessible to your project.</p>

  <h2>Data</h2>

  <p>The application expects a CSV file named `data.csv` containing song data including columns like 'Title', 'Artist', 'Album', 'Lyric', and potentially 'Date'.</p>

  <h2>Preprocessing</h2>

  <p>The `preprocessing.py` script defines a function `preprocess_text` that performs the following steps on song lyrics:</p>

  <ol>
    <li>Converts text to lowercase.</li>
    <li>Removes punctuation and special characters except for alphanumeric characters and whitespace.</li>
    <li>Tokenizes the text into words.</li>
    <li>Removes stop words from the English language (e.g., "the", "a", "is").</li>
    <li>Applies stemming to reduce words to their base forms (e.g., "running" becomes "run").</li>
  </ol>

  <h2>Core Functionality</h2>

  <ul>
    <li><strong>Embedding Generation:</strong> The `main.py` script loads the pre-trained sentence transformer model and uses it to generate embeddings for all song lyrics in the `data.csv` file. These embeddings capture the semantic similarity between lyrics.</li>
    <li><strong>Cosine Similarity:</strong> The `CosineSimilarity` class calculates the cosine similarity between two sentence embeddings. Cosine similarity ranges from -1 to 1, where higher values indicate more similar sentences.</li>
    <li><strong>Search:</strong> The `app.py` script builds the Streamlit application. When a user enters a song query, it generates an embedding for the query and calculates cosine similarities with all the song embeddings in the data.</li>
    <li><strong>Results:</strong> The application displays the most similar song based on the highest cosine similarity score. Additionally, it allows users to find the top K most similar songs by clicking a button.</li>
  </ul>

  <h2>Running the Application</h2>

  <p>1. Make sure you have the dependencies installed, the pre-trained model downloaded, and the `data.csv` file prepared.</p>
  <p>2. Navigate to the project directory in your terminal.</p>
  <p>3. Run the app using:</p>

  <pre>bash
  streamlit run app.py
  </pre>

  <p>This will launch the Streamlit app in your web browser, where you can interact with the song search interface.</p>
</body>
</html>
