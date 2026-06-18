import os
import pickle
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# LOAD MODEL
# =====================================================

print("Loading BERTopic model...")

with open(
    "models/bertopic_model.pkl",
    "rb"
) as f:
    topic_model = pickle.load(f)

# =====================================================
# LOAD TOPIC INFO
# =====================================================

topic_info = pd.read_csv(
    "data/processed/topic_info.csv"
)

# remove outlier topic
topic_info = topic_info[
    topic_info["Topic"] != -1
].reset_index(drop=True)

print(f"Topics found: {len(topic_info)}")

# =====================================================
# CREATE NODES
# =====================================================

nodes = []

for _, row in topic_info.iterrows():

    topic_id = int(row["Topic"])

    words = topic_model.get_topic(topic_id)

    if words is None:
        continue

    label = ", ".join(
        [w[0] for w in words[:5]]
    )

    nodes.append({
        "topic_id": topic_id,
        "label": label,
        "paper_count": int(row["Count"])
    })

nodes_df = pd.DataFrame(nodes)

# =====================================================
# TOPIC EMBEDDINGS
# =====================================================

topic_embeddings = topic_model.topic_embeddings_

print(
    f"Topic Embedding Shape: "
    f"{topic_embeddings.shape}"
)

similarity_matrix = cosine_similarity(
    topic_embeddings
)

# =====================================================
# CREATE EDGES
# =====================================================

SIMILARITY_THRESHOLD = 0.55

edges = []

topic_ids = nodes_df["topic_id"].tolist()

for i in range(len(topic_ids)):

    for j in range(i + 1, len(topic_ids)):

        similarity = similarity_matrix[i, j]

        if similarity >= SIMILARITY_THRESHOLD:

            edges.append({
                "source": topic_ids[i],
                "target": topic_ids[j],
                "similarity": round(
                    float(similarity), 4
                )
            })

edges_df = pd.DataFrame(edges)

# =====================================================
# SAVE
# =====================================================

os.makedirs(
    "data/graph",
    exist_ok=True
)

nodes_df.to_csv(
    "data/graph/nodes.csv",
    index=False
)

edges_df.to_csv(
    "data/graph/edges.csv",
    index=False
)

# =====================================================
# STATS
# =====================================================

print("\nGraph Dataset Created")

print(
    f"Nodes: {len(nodes_df)}"
)

print(
    f"Edges: {len(edges_df)}"
)

print("\nSample Nodes\n")
print(nodes_df.head())

print("\nSample Edges\n")
print(edges_df.head())

print(
    "\nSaved:"
)

print(
    "data/graph/nodes.csv"
)

print(
    "data/graph/edges.csv"
)