# src/topic_modeling.py

import os
import pickle
import pandas as pd

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# LOAD DATA
# ==========================================================

print("\nLoading dataset...")

df = pd.read_csv(
    "data/processed/clean_papers.csv"
)

df = df.dropna(subset=["abstract"])

documents = df["abstract"].astype(str).tolist()

print(f"Documents loaded: {len(documents)}")

# ==========================================================
# EMBEDDING MODEL
# ==========================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Generating embeddings...")

embeddings = embedding_model.encode(
    documents,
    batch_size=32,
    show_progress_bar=True
)

print(f"Embedding shape: {embeddings.shape}")

# ==========================================================
# VECTORIZER
# ==========================================================
vectorizer_model = CountVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=10000
)

# ==========================================================
# BERTOPIC
# ==========================================================

print("\nTraining BERTopic...")

topic_model = BERTopic(
    vectorizer_model=vectorizer_model,
    min_topic_size=50,
    calculate_probabilities=True,
    verbose=True
)

topics, probs = topic_model.fit_transform(
    documents,
    embeddings
)

print("\nBERTopic training complete")

# ==========================================================
# TOPIC INFO
# ==========================================================

topic_info = topic_model.get_topic_info()

print("\nTopic Summary\n")
print(topic_info.head(20))

# ==========================================================
# SAVE MODEL
# ==========================================================

os.makedirs("models", exist_ok=True)

with open(
    "models/bertopic_model.pkl",
    "wb"
) as f:

    pickle.dump(topic_model, f)

print("\nSaved BERTopic model")

# ==========================================================
# SAVE PAPER ASSIGNMENTS
# ==========================================================

df["topic"] = topics

os.makedirs(
    "data/processed",
    exist_ok=True
)

df.to_csv(
    "data/processed/papers_with_topics.csv",
    index=False
)

print("Saved papers_with_topics.csv")

# ==========================================================
# SAVE TOPIC INFO
# ==========================================================

topic_info.to_csv(
    "data/processed/topic_info.csv",
    index=False
)

print("Saved topic_info.csv")

# ==========================================================
# TOPIC EMBEDDINGS
# ==========================================================

topic_embeddings = topic_model.topic_embeddings_

print(
    f"\nTopic Embeddings Shape: "
    f"{topic_embeddings.shape}"
)

# ==========================================================
# TOPIC SIMILARITY
# ==========================================================

similarity_matrix = cosine_similarity(
    topic_embeddings
)

similarity_df = pd.DataFrame(
    similarity_matrix
)

similarity_df.to_csv(
    "data/processed/topic_similarity.csv",
    index=False
)

print("Saved topic_similarity.csv")

# ==========================================================
# DISPLAY TOPICS
# ==========================================================

print("\nTop Topics:\n")

for topic_id in topic_info["Topic"]:

    if topic_id == -1:
        continue

    words = topic_model.get_topic(
        topic_id
    )

    if words is None:
        continue

    keywords = [
        word[0]
        for word in words[:10]
    ]

    print(
        f"\nTopic {topic_id}"
    )

    print(
        ", ".join(keywords)
    )

# ==========================================================
# SAVE VISUALIZATION
# ==========================================================

os.makedirs(
    "outputs",
    exist_ok=True
)

try:

    fig = topic_model.visualize_topics()

    fig.write_html(
        "outputs/topic_visualization.html"
    )

    print(
        "\nSaved topic_visualization.html"
    )

except Exception as e:

    print(
        f"\nVisualization error: {e}"
    )

# ==========================================================
# TOP SIMILAR TOPIC PAIRS
# ==========================================================

print("\nMost Similar Topic Pairs\n")

pairs = []

num_topics = len(topic_embeddings)

for i in range(num_topics):

    for j in range(i + 1, num_topics):

        score = similarity_matrix[i, j]

        pairs.append(
            (i, j, score)
        )

pairs = sorted(
    pairs,
    key=lambda x: x[2],
    reverse=True
)

for i, j, score in pairs[:15]:

    try:

        topic_a = topic_model.get_topic(i)
        topic_b = topic_model.get_topic(j)

        if (
            topic_a is None or
            topic_b is None
        ):
            continue

        words_a = ", ".join(
            [x[0] for x in topic_a[:5]]
        )

        words_b = ", ".join(
            [x[0] for x in topic_b[:5]]
        )

        print("\n" + "=" * 70)

        print(
            f"Similarity: {score:.4f}"
        )

        print(
            f"Topic {i}: {words_a}"
        )

        print(
            f"Topic {j}: {words_b}"
        )

    except:
        continue

print("\nFinished")