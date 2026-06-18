import os
import networkx as nx
import pandas as pd
import community as community_louvain

# ====================================================
# LOAD GRAPH
# ====================================================

print("Loading graph...")

nodes_df = pd.read_csv(
    "data/graph/nodes.csv"
)

edges_df = pd.read_csv(
    "data/graph/edges.csv"
)

# ====================================================
# BUILD GRAPH
# ====================================================

G = nx.Graph()

# Nodes
for _, row in nodes_df.iterrows():

    G.add_node(
        int(row["topic_id"]),
        label=row["label"]
    )

# Edges
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

# ====================================================
# COMMUNITY DETECTION
# ====================================================

print("\nRunning Louvain community detection...")

partition = community_louvain.best_partition(
    G,
    weight="weight"
)

# ====================================================
# CREATE DATAFRAME
# ====================================================

community_df = pd.DataFrame(
    {
        "topic_id": list(partition.keys()),
        "community": list(partition.values())
    }
)

# ====================================================
# MERGE LABELS
# ====================================================

community_df = community_df.merge(
    nodes_df[["topic_id", "label"]],
    on="topic_id"
)

# ====================================================
# SAVE
# ====================================================

os.makedirs(
    "outputs",
    exist_ok=True
)

community_df.to_csv(
    "outputs/topic_communities.csv",
    index=False
)

print()

print("Communities Found:",
      community_df["community"].nunique())

print()

# ====================================================
# DISPLAY COMMUNITIES
# ====================================================

for community_id in sorted(
        community_df["community"].unique()):

    print("=" * 70)

    print(
        f"Community {community_id}"
    )

    temp = community_df[
        community_df["community"] == community_id
    ]

    for _, row in temp.iterrows():

        print(
            f"Topic {row['topic_id']} : {row['label']}"
        )

print()

print(
    "Saved outputs/topic_communities.csv"
)