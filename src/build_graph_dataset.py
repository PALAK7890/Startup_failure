# src/build_graph_dataset.py

import os
import pandas as pd

from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# DIRECTORIES
# ==========================================================

os.makedirs(
    "data/graph",
    exist_ok=True
)

# ==========================================================
# LOAD MODEL
# ==========================================================

print("Loading BERTopic model...")

topic_model = BERTopic.load(
    "models/bertopic_model"
)

# ==========================================================
# LOAD TOPIC INFO
# ==========================================================

topic_info = pd.read_csv(
    "data/processed/topic_info.csv"
)

# remove outlier topic
topic_info = topic_info[
    topic_info["Topic"] != -1
].reset_index(drop=True)

valid_topic_ids = (
    topic_info["Topic"]
    .tolist()
)

print(
    f"Topics found: {len(valid_topic_ids)}"
)

# ==========================================================
# CREATE NODES
# ==========================================================

nodes = []

for _, row in topic_info.iterrows():

    topic_id = int(
        row["Topic"]
    )

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

    nodes.append(
        {
            "topic_id": topic_id,
            "label": label,
            "paper_count": int(
                row["Count"]
            )
        }
    )

nodes_df = pd.DataFrame(
    nodes
)

# ==========================================================
# TOPIC EMBEDDINGS
# ==========================================================

topic_embeddings = (
    topic_model.topic_embeddings_
)

# remove topic -1
topic_embeddings = topic_embeddings[
    valid_topic_ids
]

print(
    "Topic Embedding Shape:",
    topic_embeddings.shape
)

# ==========================================================
# SIMILARITY MATRIX
# ==========================================================

similarity_matrix = cosine_similarity(
    topic_embeddings
)

# ==========================================================
# CREATE EDGES
# ==========================================================

SIMILARITY_THRESHOLD = 0.70

edges = []

for i in range(
    len(valid_topic_ids)
):

    for j in range(
        i + 1,
        len(valid_topic_ids)
    ):

        similarity = similarity_matrix[
            i,
            j
        ]

        if similarity >= SIMILARITY_THRESHOLD:

            edges.append(
                {
                    "source":
                    valid_topic_ids[i],

                    "target":
                    valid_topic_ids[j],

                    "similarity":
                    round(
                        float(
                            similarity
                        ),
                        4
                    )
                }
            )

edges_df = pd.DataFrame(
    edges
)

# ==========================================================
# SAVE
# ==========================================================

nodes_df.to_csv(
    "data/graph/nodes.csv",
    index=False
)

edges_df.to_csv(
    "data/graph/edges.csv",
    index=False
)

# ==========================================================
# SUMMARY
# ==========================================================

print("\nGraph Dataset Created")

print(
    "Nodes:",
    len(nodes_df)
)

print(
    "Edges:",
    len(edges_df)
)

print("\nSample Nodes\n")

print(
    nodes_df.head()
)

print("\nSample Edges\n")

print(
    edges_df.head()
)

print("\nSaved")

print(
    "data/graph/nodes.csv"
)

print(
    "data/graph/edges.csv"
)