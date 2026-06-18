# src/topic_modeling.py

import os
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from umap import UMAP
from hdbscan import HDBSCAN


# ==========================================================
# CREATE DIRECTORIES
# ==========================================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/graph", exist_ok=True)


# ==========================================================
# LOAD DATA
# ==========================================================

print("\nLoading dataset...")

df = pd.read_csv(
    "data/processed/clean_papers.csv"
)

df = df.dropna(
    subset=["abstract"]
)

documents = (
    df["abstract"]
    .astype(str)
    .tolist()
)

print(
    f"Documents loaded: {len(documents)}"
)


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

print(
    "Embedding shape:",
    embeddings.shape
)


# ==========================================================
# UMAP
# ==========================================================

umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42
)


# ==========================================================
# HDBSCAN
# ==========================================================

hdbscan_model = HDBSCAN(
    min_cluster_size=8,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True
)


# ==========================================================
# VECTORIZER
# ==========================================================

vectorizer_model = CountVectorizer(
    stop_words="english",
    ngram_range=(1,2),
    min_df=2,
    max_features=10000
)


# ==========================================================
# BERTOPIC MODEL
# ==========================================================

print("\nTraining BERTopic...")

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    min_topic_size=8,
    calculate_probabilities=True,
    verbose=True
)


# ==========================================================
# FIT MODEL
# ==========================================================

topics, probs = topic_model.fit_transform(
    documents,
    embeddings
)

print("\nBERTopic training complete")


# ==========================================================
# SAVE MODEL
# ==========================================================

topic_model.save(
    "models/bertopic_model"
)

print("Saved BERTopic model")

# ==========================================================
# TOPIC INFORMATION
# ==========================================================

topic_info = topic_model.get_topic_info()

print("\nTopic Summary\n")

print(
    topic_info.head(20)
)


# ==========================================================
# SAVE TOPIC INFO
# ==========================================================

topic_info.to_csv(
    "data/processed/topic_info.csv",
    index=False
)

print(
    "Saved topic_info.csv"
)


# ==========================================================
# SAVE PAPER TOPICS
# ==========================================================

df["topic"] = topics

df.to_csv(
    "data/processed/papers_with_topics.csv",
    index=False
)

print(
    "Saved papers_with_topics.csv"
)


# ==========================================================
# PAPER-TOPIC MAPPING
# ==========================================================

paper_topic_mapping = pd.DataFrame(
    {
        "paper_id": df.index,
        "topic": topics
    }
)

paper_topic_mapping.to_csv(
    "data/processed/paper_topic_mapping.csv",
    index=False
)

print(
    "Saved paper_topic_mapping.csv"
)


# ==========================================================
# REMOVE OUTLIER TOPIC (-1)
# ==========================================================

valid_topics = topic_info[
    topic_info["Topic"] != -1
]

valid_topic_ids = (
    valid_topics["Topic"]
    .tolist()
)

print(
    "\nValid topics:",
    len(valid_topic_ids)
)


# ==========================================================
# TOPIC EMBEDDINGS
# ==========================================================

topic_embeddings = topic_model.topic_embeddings_

topic_embeddings = topic_embeddings[
    valid_topic_ids
]

print(
    "Topic Embeddings Shape:",
    topic_embeddings.shape
)


# ==========================================================
# SAVE TOPIC EMBEDDINGS
# ==========================================================

topic_embeddings_df = pd.DataFrame(
    topic_embeddings
)

topic_embeddings_df.insert(
    0,
    "topic_id",
    valid_topic_ids
)

topic_embeddings_df.to_csv(
    "data/processed/topic_embeddings.csv",
    index=False
)

print(
    "Saved topic_embeddings.csv"
)
# ==========================================================
# TOPIC SIMILARITY MATRIX
# ==========================================================

print("\nComputing topic similarity matrix...")

similarity_matrix = cosine_similarity(
    topic_embeddings
)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=valid_topic_ids,
    columns=valid_topic_ids
)

similarity_df.to_csv(
    "data/processed/topic_similarity.csv"
)

print("Saved topic_similarity.csv")


# ==========================================================
# BUILD GRAPH EDGES
# ==========================================================

SIMILARITY_THRESHOLD = 0.70

edges = []

for i in range(len(valid_topic_ids)):

    for j in range(i + 1, len(valid_topic_ids)):

        similarity_score = similarity_matrix[i, j]

        if similarity_score >= SIMILARITY_THRESHOLD:

            edges.append(
                [
                    valid_topic_ids[i],
                    valid_topic_ids[j],
                    similarity_score
                ]
            )

edges_df = pd.DataFrame(
    edges,
    columns=[
        "source",
        "target",
        "similarity"
    ]
)

edges_df.to_csv(
    "data/graph/edges.csv",
    index=False
)

print(
    f"Saved edges.csv ({len(edges_df)} edges)"
)


# ==========================================================
# CREATE TOPIC LABELS
# ==========================================================

topic_labels = []

for topic_id in valid_topic_ids:

    words = topic_model.get_topic(
        topic_id
    )

    if words is None:
        continue

    label = "_".join(
        [
            x[0]
            for x in words[:3]
        ]
    )

    topic_labels.append(
        [
            topic_id,
            label
        ]
    )

topic_labels_df = pd.DataFrame(
    topic_labels,
    columns=[
        "topic_id",
        "label"
    ]
)

topic_labels_df.to_csv(
    "data/graph/topic_labels.csv",
    index=False
)

print(
    "Saved topic_labels.csv"
)


# ==========================================================
# CREATE NODES
# ==========================================================

node_counts = (
    df["topic"]
    .value_counts()
)

nodes = []

for topic_id in valid_topic_ids:

    words = topic_model.get_topic(
        topic_id
    )

    if words is None:
        continue

    label = ", ".join(
        [
            x[0]
            for x in words[:5]
        ]
    )

    count = int(
        node_counts.get(
            topic_id,
            0
        )
    )

    nodes.append(
        [
            topic_id,
            label,
            count
        ]
    )

nodes_df = pd.DataFrame(
    nodes,
    columns=[
        "topic_id",
        "label",
        "paper_count"
    ]
)

nodes_df.to_csv(
    "data/graph/nodes.csv",
    index=False
)

print(
    f"Saved nodes.csv ({len(nodes_df)} nodes)"
)
# ==========================================================
# DISPLAY TOPICS
# ==========================================================

print("\nTop Topics\n")

for topic_id in valid_topic_ids:

    words = topic_model.get_topic(
        topic_id
    )

    if words is None:
        continue

    keywords = [
        x[0]
        for x in words[:10]
    ]

    print("\nTopic", topic_id)

    print(
        ", ".join(keywords)
    )


# ==========================================================
# HIERARCHICAL TOPICS
# ==========================================================

print("\nGenerating hierarchical topics...")

try:

    hierarchical_topics = (
        topic_model.hierarchical_topics(
            documents
        )
    )

    hierarchical_topics.to_csv(
        "data/processed/hierarchical_topics.csv",
        index=False
    )

    print(
        "Saved hierarchical_topics.csv"
    )

except Exception as e:

    print(
        f"Hierarchy error: {e}"
    )


# ==========================================================
# TOPIC VISUALIZATION
# ==========================================================

print("\nGenerating topic visualization...")

try:

    fig = topic_model.visualize_topics()

    fig.write_html(
        "outputs/topic_visualization.html"
    )

    print(
        "Saved topic_visualization.html"
    )

except Exception as e:

    print(
        f"Visualization error: {e}"
    )


# ==========================================================
# HIERARCHY VISUALIZATION
# ==========================================================

try:

    hierarchy_fig = (
        topic_model.visualize_hierarchy()
    )

    hierarchy_fig.write_html(
        "outputs/topic_hierarchy.html"
    )

    print(
        "Saved topic_hierarchy.html"
    )

except Exception as e:

    print(
        f"Hierarchy visualization error: {e}"
    )


# ==========================================================
# HEATMAP
# ==========================================================

try:

    heatmap_fig = (
        topic_model.visualize_heatmap()
    )

    heatmap_fig.write_html(
        "outputs/topic_heatmap.html"
    )

    print(
        "Saved topic_heatmap.html"
    )

except Exception as e:

    print(
        f"Heatmap error: {e}"
    )


# ==========================================================
# TOP SIMILAR TOPIC PAIRS
# ==========================================================

print("\nMost Similar Topic Pairs\n")

pairs = []

n_topics = len(valid_topic_ids)

for i in range(n_topics):

    for j in range(i + 1, n_topics):

        score = similarity_matrix[i, j]

        pairs.append(
            (
                valid_topic_ids[i],
                valid_topic_ids[j],
                score
            )
        )

pairs = sorted(
    pairs,
    key=lambda x: x[2],
    reverse=True
)

top_pairs = []

for topic_a, topic_b, score in pairs[:20]:

    words_a = topic_model.get_topic(
        topic_a
    )

    words_b = topic_model.get_topic(
        topic_b
    )

    if (
        words_a is None
        or
        words_b is None
    ):
        continue

    label_a = ", ".join(
        [
            x[0]
            for x in words_a[:5]
        ]
    )

    label_b = ", ".join(
        [
            x[0]
            for x in words_b[:5]
        ]
    )

    top_pairs.append(
        [
            topic_a,
            topic_b,
            score,
            label_a,
            label_b
        ]
    )

    print("\n" + "=" * 70)

    print(
        f"Similarity: {score:.4f}"
    )

    print(
        f"Topic {topic_a}: {label_a}"
    )

    print(
        f"Topic {topic_b}: {label_b}"
    )


# ==========================================================
# SAVE TOP PAIRS
# ==========================================================

top_pairs_df = pd.DataFrame(
    top_pairs,
    columns=[
        "topic_a",
        "topic_b",
        "similarity",
        "label_a",
        "label_b"
    ]
)

top_pairs_df.to_csv(
    "data/processed/top_similar_topic_pairs.csv",
    index=False
)

print(
    "\nSaved top_similar_topic_pairs.csv"
)


# ==========================================================
# FINISHED
# ==========================================================

print("\n" + "=" * 80)
print("TOPIC MODELING PIPELINE COMPLETE")
print("=" * 80)

print("\nGenerated files:")

print("""
models/
    bertopic_model/

data/processed/
    papers_with_topics.csv
    paper_topic_mapping.csv
    topic_info.csv
    topic_embeddings.csv
    topic_similarity.csv
    hierarchical_topics.csv
    top_similar_topic_pairs.csv

data/graph/
    nodes.csv
    edges.csv
    topic_labels.csv

outputs/
    topic_visualization.html
    topic_hierarchy.html
    topic_heatmap.html
""")