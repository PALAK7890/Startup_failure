import os
import pandas as pd
from pyvis.network import Network

# =====================================================
# CREATE OUTPUT DIRECTORY
# =====================================================

os.makedirs(
    "outputs",
    exist_ok=True
)

# =====================================================
# LOAD DATA
# =====================================================

print("Loading graph data...")

nodes_df = pd.read_csv(
    "data/graph/nodes.csv"
)

edges_df = pd.read_csv(
    "data/graph/edges.csv"
)

print(
    f"Nodes: {len(nodes_df)}"
)

print(
    f"Edges: {len(edges_df)}"
)

# =====================================================
# CREATE NETWORK
# =====================================================

net = Network(
    height="900px",
    width="100%",
    bgcolor="#222222",
    font_color="white",
    notebook=False
)

# =====================================================
# PHYSICS
# =====================================================

net.force_atlas_2based(
    gravity=-50,
    central_gravity=0.01,
    spring_length=120,
    spring_strength=0.08,
    damping=0.4
)

# =====================================================
# ADD NODES
# =====================================================

print("Adding nodes...")

for _, row in nodes_df.iterrows():

    topic_id = int(row["topic_id"])

    label = str(row["label"])

    paper_count = int(row["paper_count"])

    tooltip = f"""
    <b>Topic {topic_id}</b><br>
    Papers: {paper_count}<br><br>
    {label}
    """

    net.add_node(
        n_id=topic_id,
        label=str(topic_id),
        title=tooltip,
        value=paper_count,
        color="#00bfff"
    )

# =====================================================
# ADD EDGES
# =====================================================

print("Adding edges...")

for _, row in edges_df.iterrows():

    source = int(row["source"])

    target = int(row["target"])

    similarity = float(row["similarity"])

    net.add_edge(
        source,
        target,
        value=similarity * 10,
        title=f"Similarity: {similarity:.3f}",
        color="#888888"
    )

# =====================================================
# OPTIONS
# =====================================================

net.set_options("""
var options = {
  "nodes": {
    "shape": "dot",
    "font": {
      "size": 16
    }
  },
  "edges": {
    "smooth": false
  },
  "physics": {
    "enabled": true,
    "forceAtlas2Based": {
      "gravitationalConstant": -50,
      "centralGravity": 0.01,
      "springLength": 120,
      "springConstant": 0.08
    },
    "solver": "forceAtlas2Based"
  },
  "interaction": {
    "hover": true,
    "navigationButtons": true,
    "keyboard": true
  }
}
""")

# =====================================================
# SAVE GRAPH
# =====================================================

output_file = "outputs/topic_graph.html"

net.show(
    output_file
)

print()

print("=" * 50)
print("GRAPH VISUALIZATION CREATED")
print("=" * 50)

print()

print("Saved:")

print(output_file)

print()

print("Open in browser:")
print(output_file)