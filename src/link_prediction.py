import pandas as pd
import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from sklearn.model_selection import train_test_split

# ======================================
# LOAD FEATURES
# ======================================

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

# ======================================
# LOAD EDGES
# ======================================

edges_df = pd.read_csv(
    "data/graph/edges.csv"
)

positive_edges = list(
    zip(
        edges_df["source"],
        edges_df["target"]
    )
)

# ======================================
# NEGATIVE SAMPLING
# ======================================

num_nodes = len(features_df)

positive_set = set(
    tuple(sorted(edge))
    for edge in positive_edges
)

negative_edges = []

for i in range(num_nodes):

    for j in range(i + 1, num_nodes):

        if (i, j) not in positive_set:

            negative_edges.append(
                (i, j)
            )

negative_edges = negative_edges[
    :len(positive_edges)
]

# ======================================
# TRAIN DATA
# ======================================

all_edges = (
    positive_edges +
    negative_edges
)

labels = (
    [1] * len(positive_edges) +
    [0] * len(negative_edges)
)

train_edges, test_edges, train_labels, test_labels = train_test_split(
    all_edges,
    labels,
    test_size=0.2,
    random_state=42
)

# ======================================
# GRAPH
# ======================================

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

# ======================================
# MODEL
# ======================================

class LinkPredictor(
    torch.nn.Module
):

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

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(
                128,
                64
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                64,
                1
            )
        )

    def encode(
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

    def decode(
        self,
        z,
        edges
    ):

        edge_embeddings = []

        for src, dst in edges:

            edge_embeddings.append(
                torch.cat(
                    [
                        z[src],
                        z[dst]
                    ]
                )
            )

        edge_embeddings = torch.stack(
            edge_embeddings
        )

        return self.classifier(
            edge_embeddings
        ).squeeze()

# ======================================
# TRAIN
# ======================================

model = LinkPredictor()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

loss_fn = torch.nn.BCEWithLogitsLoss()

for epoch in range(100):

    model.train()

    optimizer.zero_grad()

    z = model.encode(
        data.x,
        data.edge_index
    )

    preds = model.decode(
        z,
        train_edges
    )

    y = torch.tensor(
        train_labels,
        dtype=torch.float
    )

    loss = loss_fn(
        preds,
        y
    )

    loss.backward()

    optimizer.step()

    if epoch % 10 == 0:

        print(
            f"Epoch {epoch} | Loss {loss.item():.4f}"
        )

# ======================================
# TEST
# ======================================

model.eval()

with torch.no_grad():

    z = model.encode(
        data.x,
        data.edge_index
    )

    preds = torch.sigmoid(
        model.decode(
            z,
            test_edges
        )
    )

print("\nPredictions\n")

for edge, score in zip(
    test_edges,
    preds
):

    print(
        edge,
        round(
            score.item(),
            3
        )
    )