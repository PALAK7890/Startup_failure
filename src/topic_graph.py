import pickle
import pandas as pd
import networkx as nx

from sklearn.metrics.pairwise import cosine_similarity
from pyvis.network import Network

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

topic_info = topic_model.get_topic_info()

topic_info = topic_info[
    topic_info["Topic"] != -1
]

print(f"Topics Found: {len(topic_info)}")

# =====================================================
# TOPIC EMBEDDINGS
# =====================================================

topic_embeddings = topic_model.topic_embeddings_

print(
    f"Embedding Shape: "
    f"{topic_embeddings.shape}"
)

# =====================================================
# SIMILARITY MATRIX
# =====================================================

similarity_matrix = cosine_similarity(
    topic_embeddings
)

# =====================================================
# CREATE GRAPH
# =====================================================

G = nx.Graph()

# =====================================================
# ADD NODES
# =====================================================

for topic_id in topic_info["Topic"]:

    words = topic_model.get_topic(topic_id)

    if words is None:
        continue

    label = ", ".join(
        [w[0] for w in words[:3]]
    )

    count = int(
        topic_info.loc[
            topic_info["Topic"] == topic_id,
            "Count"
        ].values[0]
    )

    G.add_node(
        topic_id,
        label=label,
        size=count
    )

# =====================================================
# ADD EDGES
# =====================================================

SIMILARITY_THRESHOLD = 0.70

num_topics = len(topic_embeddings)

for i in range(num_topics):

    for j in range(i + 1, num_topics):

        similarity = similarity_matrix[i, j]

        if similarity >= SIMILARITY_THRESHOLD:

            G.add_edge(
                i,
                j,
                weight=float(similarity)
            )

# =====================================================
# GRAPH STATS
# =====================================================

print("\nGraph Statistics")

print(
    f"Nodes: {G.number_of_nodes()}"
)

print(
    f"Edges: {G.number_of_edges()}"
)

# =====================================================
# SAVE EDGE LIST
# =====================================================

edges = []

for u, v, data in G.edges(data=True):

    edges.append({
        "topic_a": u,
        "topic_b": v,
        "similarity": data["weight"]
    })

edges_df = pd.DataFrame(edges)

edges_df.to_csv(
    "outputs/topic_edges.csv",
    index=False
)

print(
    "Saved outputs/topic_edges.csv"
)

# =====================================================
# NETWORK METRICS
# =====================================================

centrality = nx.degree_centrality(G)

centrality_df = pd.DataFrame({
    "topic": list(
        centrality.keys()
    ),
    "centrality": list(
        centrality.values()
    )
})

centrality_df = centrality_df.sort_values(
    by="centrality",
    ascending=False
)

centrality_df.to_csv(
    "outputs/topic_centrality.csv",
    index=False
)

print(
    "Saved outputs/topic_centrality.csv"
)

# =====================================================
# INTERACTIVE VISUALIZATION
# =====================================================

net = Network(
    height="800px",
    width="100%",
    bgcolor="#ffffff",
    font_color="black"
)

for node, attrs in G.nodes(data=True):

    net.add_node(
        node,
        label=attrs["label"],
        size=max(
            attrs["size"] / 20,
            10
        )
    )

for source, target, attrs in G.edges(data=True):

    net.add_edge(
        source,
        target,
        value=attrs["weight"]
    )

net.write_html(
    "outputs/topic_graph.html"
)

print(
    "Saved outputs/topic_graph.html"
)

print("\nDone!")