import os
import networkx as nx
import pandas as pd

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
        int(row["topic_id"])
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
# CENTRALITY METRICS
# ====================================================

print("\nComputing graph metrics...")

degree_centrality = nx.degree_centrality(
    G
)

betweenness_centrality = nx.betweenness_centrality(
    G
)

closeness_centrality = nx.closeness_centrality(
    G
)

eigenvector_centrality = nx.eigenvector_centrality(
    G,
    max_iter=1000
)

pagerank = nx.pagerank(
    G
)

clustering_coefficient = nx.clustering(
    G
)

# ====================================================
# CREATE DATAFRAME
# ====================================================

metrics_df = pd.DataFrame({

    "topic_id": list(
        degree_centrality.keys()
    ),

    "degree_centrality": list(
        degree_centrality.values()
    ),

    "betweenness_centrality": [
        betweenness_centrality[n]
        for n in degree_centrality
    ],

    "closeness_centrality": [
        closeness_centrality[n]
        for n in degree_centrality
    ],

    "eigenvector_centrality": [
        eigenvector_centrality[n]
        for n in degree_centrality
    ],

    "pagerank": [
        pagerank[n]
        for n in degree_centrality
    ],

    "clustering_coefficient": [
        clustering_coefficient[n]
        for n in degree_centrality
    ]

})

# ====================================================
# MERGE LABELS
# ====================================================

metrics_df = metrics_df.merge(
    nodes_df[
        ["topic_id", "label"]
    ],
    on="topic_id"
)

# ====================================================
# SAVE
# ====================================================

os.makedirs(
    "outputs",
    exist_ok=True
)

metrics_df.to_csv(
    "outputs/graph_metrics.csv",
    index=False
)

# ====================================================
# GRAPH STATISTICS
# ====================================================

print("\nGRAPH STATISTICS\n")

print(
    "Density:",
    round(
        nx.density(G),
        4
    )
)

print(
    "Average Clustering Coefficient:",
    round(
        nx.average_clustering(G),
        4
    )
)

print(
    "Connected Components:",
    nx.number_connected_components(G)
)

# ====================================================
# TOP PAGERANK TOPICS
# ====================================================

print("\nTop Topics by PageRank\n")

top_topics = metrics_df.sort_values(
    by="pagerank",
    ascending=False
)

for _, row in top_topics.head(10).iterrows():

    print("=" * 80)

    print(
        f"Topic {row['topic_id']}"
    )

    print(
        row["label"]
    )

    print()

    print(
        "PageRank:",
        round(
            row["pagerank"],
            4
        )
    )

# ====================================================
# TOP BETWEENNESS
# ====================================================

print("\nTop Topics by Betweenness\n")

top_topics = metrics_df.sort_values(
    by="betweenness_centrality",
    ascending=False
)

for _, row in top_topics.head(10).iterrows():

    print("=" * 80)

    print(
        f"Topic {row['topic_id']}"
    )

    print(
        row["label"]
    )

    print()

    print(
        "Betweenness:",
        round(
            row["betweenness_centrality"],
            4
        )
    )

print()

print(
    "Saved outputs/graph_metrics.csv"
)