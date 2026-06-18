import os
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from node2vec import Node2Vec
from sklearn.manifold import TSNE

# ===================================================
# LOAD DATA
# ===================================================

print("Loading graph...")

nodes_df = pd.read_csv(
    "data/graph/nodes.csv"
)

edges_df = pd.read_csv(
    "data/graph/edges.csv"
)

# ===================================================
# BUILD GRAPH
# ===================================================

G = nx.Graph()

# Add nodes
for _, row in nodes_df.iterrows():

    G.add_node(
        int(row["topic_id"])
    )

# Add edges
for _, row in edges_df.iterrows():

    G.add_edge(
        int(row["source"]),
        int(row["target"]),
        weight=float(row["similarity"])
    )

print(
    f"Nodes: {G.number_of_nodes()}"
)

print(
    f"Edges: {G.number_of_edges()}"
)

# ===================================================
# NODE2VEC
# ===================================================

print("\nTraining Node2Vec...")

node2vec = Node2Vec(
    G,
    dimensions=128,
    walk_length=20,
    num_walks=200,
    workers=4,
    seed=42
)

model = node2vec.fit(
    window=10,
    min_count=1
)

print("Node2Vec complete")

# ===================================================
# EMBEDDINGS
# ===================================================

embeddings = []

node_ids = []

for node in G.nodes():

    embeddings.append(
        model.wv[str(node)]
    )

    node_ids.append(
        node
    )

embeddings = np.array(
    embeddings
)

print(
    "Embedding shape:",
    embeddings.shape
)

# ===================================================
# TSNE
# ===================================================

print("\nRunning t-SNE...")

tsne = TSNE(
    n_components=2,
    perplexity=10,
    random_state=42
)

emb_2d = tsne.fit_transform(
    embeddings
)

# ===================================================
# PLOT
# ===================================================

plt.figure(
    figsize=(12,10)
)

plt.scatter(
    emb_2d[:,0],
    emb_2d[:,1],
    alpha=0.8
)

# Labels
for i, topic_id in enumerate(node_ids):

    plt.text(
        emb_2d[i,0],
        emb_2d[i,1],
        str(topic_id),
        fontsize=8
    )

plt.title(
    "Node2Vec + t-SNE Topic Graph"
)

plt.xlabel(
    "t-SNE Dimension 1"
)

plt.ylabel(
    "t-SNE Dimension 2"
)

plt.tight_layout()

# ===================================================
# SAVE
# ===================================================

os.makedirs(
    "outputs",
    exist_ok=True
)

plt.savefig(
    "outputs/node2vec_tsne.png",
    dpi=300
)

plt.show()

print()

print(
    "Saved outputs/node2vec_tsne.png"
)