import os
import networkx as nx
import pandas as pd
import numpy as np

from bertopic import BERTopic

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

topic_info = topic_info[
    topic_info["Topic"] != -1
].reset_index(drop=True)

valid_topic_ids = (
    topic_info["Topic"]
    .tolist()
)

# ==========================================================
# LOAD GRAPH
# ==========================================================

edges_df = pd.read_csv(
    "data/graph/edges.csv"
)

nodes_df = pd.read_csv(
    "data/graph/nodes.csv"
)

# ==========================================================
# BUILD NETWORKX GRAPH
# ==========================================================

G = nx.Graph()

for _, row in nodes_df.iterrows():

    G.add_node(
        row["topic_id"]
    )

for _, row in edges_df.iterrows():

    G.add_edge(
        row["source"],
        row["target"],
        weight=row["similarity"]
    )

# ==========================================================
# GRAPH FEATURES
# ==========================================================

degree_centrality = nx.degree_centrality(
    G
)

closeness_centrality = nx.closeness_centrality(
    G
)

betweenness_centrality = nx.betweenness_centrality(
    G
)

clustering_coeff = nx.clustering(
    G
)

pagerank = nx.pagerank(
    G
)

# ==========================================================
# TOPIC EMBEDDINGS
# ==========================================================

topic_embeddings = (
    topic_model.topic_embeddings_
)

topic_embeddings = topic_embeddings[
    valid_topic_ids
]

# ==========================================================
# CREATE FEATURE MATRIX
# ==========================================================

features = []

for i, topic_id in enumerate(valid_topic_ids):

    embedding = topic_embeddings[i]

    feature_row = {}

    feature_row["topic_id"] = topic_id

    # ------------------------------------
    # BERTopic embedding
    # ------------------------------------
    for j, value in enumerate(embedding):

        feature_row[f"emb_{j}"] = value

    # ------------------------------------
    # paper count
    # ------------------------------------
    count = topic_info.loc[
        topic_info["Topic"] == topic_id,
        "Count"
    ].values[0]

    feature_row["paper_count"] = count

    # ------------------------------------
    # degree centrality
    # ------------------------------------
    feature_row["degree_centrality"] = (
        degree_centrality[topic_id]
    )

    # ------------------------------------
    # closeness
    # ------------------------------------
    feature_row["closeness_centrality"] = (
        closeness_centrality[topic_id]
    )

    # ------------------------------------
    # betweenness
    # ------------------------------------
    feature_row["betweenness_centrality"] = (
        betweenness_centrality[topic_id]
    )

    # ------------------------------------
    # clustering coefficient
    # ------------------------------------
    feature_row["clustering_coeff"] = (
        clustering_coeff[topic_id]
    )

    # ------------------------------------
    # pagerank
    # ------------------------------------
    feature_row["pagerank"] = (
        pagerank[topic_id]
    )

    # ------------------------------------
    # average similarity
    # ------------------------------------
    neighbors = list(
        G.neighbors(topic_id)
    )

    if len(neighbors) > 0:

        similarities = []

        for n in neighbors:

            similarities.append(
                G[topic_id][n]["weight"]
            )

        avg_similarity = np.mean(
            similarities
        )

    else:

        avg_similarity = 0

    feature_row["avg_similarity"] = (
        avg_similarity
    )

    features.append(
        feature_row
    )

# ==========================================================
# SAVE
# ==========================================================

features_df = pd.DataFrame(
    features
)

os.makedirs(
    "data/graph",
    exist_ok=True
)

features_df.to_csv(
    "data/graph/node_features.csv",
    index=False
)

# ==========================================================
# SUMMARY
# ==========================================================

print()

print(
    "Feature matrix shape:",
    features_df.shape
)

print()

print(
    features_df.head()
)

print()

print(
    "Saved node_features.csv"
)