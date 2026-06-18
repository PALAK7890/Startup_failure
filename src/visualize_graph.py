import pandas as pd
import networkx as nx

from pyvis.network import Network

# ===================================================
# LOAD DATA
# ===================================================

nodes_df = pd.read_csv(
    "data/graph/nodes.csv"
)

edges_df = pd.read_csv(
    "data/graph/edges.csv"
)

# ===================================================
# CREATE GRAPH
# ===================================================

G = nx.Graph()

# ----------------------------
# ADD NODES
# ----------------------------

for _, row in nodes_df.iterrows():

    G.add_node(
        row["topic_id"],
        label=row["label"],
        size=row["paper_count"]
    )

# ----------------------------
# ADD EDGES
# ----------------------------

for _, row in edges_df.iterrows():

    G.add_edge(
        row["source"],
        row["target"],
        weight=row["similarity"]
    )

# ===================================================
# PYVIS
# ===================================================

net = Network(
    height="900px",
    width="100%",
    bgcolor="#222222",
    font_color="white",
    notebook=False
)

net.from_nx(G)

# ===================================================
# NODE STYLING
# ===================================================

for node in net.nodes:

    topic_id = node["id"]

    count = nodes_df.loc[
        nodes_df["topic_id"] == topic_id,
        "paper_count"
    ].values[0]

    label = nodes_df.loc[
        nodes_df["topic_id"] == topic_id,
        "label"
    ].values[0]

    node["title"] = (
        f"Topic ID: {topic_id}<br>"
        f"Papers: {count}<br><br>"
        f"{label}"
    )

    node["value"] = count

# ===================================================
# EDGE STYLING
# ===================================================

for edge in net.edges:

    similarity = G[
        edge["from"]
    ][
        edge["to"]
    ]["weight"]

    edge["title"] = (
        f"Similarity: {similarity:.3f}"
    )

    edge["value"] = similarity * 10

# ===================================================
# PHYSICS
# ===================================================

net.force_atlas_2based(
    gravity=-50,
    central_gravity=0.01,
    spring_length=120,
    spring_strength=0.08
)

# ===================================================
# SAVE
# ===================================================

net.save_graph(
    "outputs/topic_graph.html"
)

print()

print("Saved:")

print(
    "outputs/topic_graph.html"
)