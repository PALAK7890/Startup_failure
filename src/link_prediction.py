import pandas as pd
import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# =====================================
# LOAD FEATURES
# =====================================

features_df = pd.read_csv(
    "data/graph/node_features.csv"
)

features_df = features_df.sort_values(
    "topic_id"
)

X = torch.tensor(
    features_df.drop(
        columns=["topic_id"]
    ).values,
    dtype=torch.float
)

# =====================================
# LOAD EDGES
# =====================================

edges_df = pd.read_csv(
    "data/graph/edges.csv"
)

edge_index = torch.tensor(
    [
        edges_df["source"].tolist(),
        edges_df["target"].tolist()
    ],
    dtype=torch.long
)

edge_index = torch.cat(
    [
        edge_index,
        edge_index.flip(0)
    ],
    dim=1
)

data = Data(
    x=X,
    edge_index=edge_index
)

# =====================================
# GCN
# =====================================

class GCNEncoder(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = GCNConv(
            384,
            128
        )

        self.conv2 = GCNConv(
            128,
            64
        )

    def forward(
        self,
        x,
        edge_index
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        return x

model = GCNEncoder()

embeddings = model(
    data.x,
    data.edge_index
)

print(
    "Node Embeddings Shape:"
)

print(
    embeddings.shape
)