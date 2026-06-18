# src/link_prediction.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import SAGEConv


# ====================================================
# LOAD NODE FEATURES
# ====================================================

print("Loading node features...")

features_df = pd.read_csv(
    "data/graph/node_features.csv"
)

features_df = features_df.sort_values(
    "topic_id"
)

X = features_df.drop(
    columns=["topic_id"]
)

# Scale features
scaler = StandardScaler()

X = scaler.fit_transform(X)

X = torch.tensor(
    X,
    dtype=torch.float
)

input_dim = X.shape[1]

print(
    "Feature shape:",
    X.shape
)


# ====================================================
# LOAD GRAPH EDGES
# ====================================================

print("Loading graph...")

edges_df = pd.read_csv(
    "data/graph/edges.csv"
)

edge_index = np.array(
    [
        edges_df["source"].values,
        edges_df["target"].values
    ]
)

edge_index = torch.tensor(
    edge_index,
    dtype=torch.long
)

# Undirected graph
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

print(data)


# ====================================================
# RANDOM LINK SPLIT
# ====================================================

transform = RandomLinkSplit(
    num_val=0.1,
    num_test=0.2,
    is_undirected=True,
    add_negative_train_samples=True,
    neg_sampling_ratio=1.0
)

train_data, val_data, test_data = transform(
    data
)

print()

print(
    "Train edges:",
    train_data.edge_label.shape[0]
)

print(
    "Val edges:",
    val_data.edge_label.shape[0]
)

print(
    "Test edges:",
    test_data.edge_label.shape[0]
)


# ====================================================
# MODEL
# ====================================================

class GraphSAGELinkPredictor(
    nn.Module
):

    def __init__(
        self,
        input_dim
    ):

        super().__init__()

        self.conv1 = SAGEConv(
            input_dim,
            128
        )

        self.bn1 = nn.BatchNorm1d(
            128
        )

        self.conv2 = SAGEConv(
            128,
            64
        )

        self.bn2 = nn.BatchNorm1d(
            64
        )

        self.dropout = nn.Dropout(
            0.3
        )

        self.decoder = nn.Sequential(

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                64,
                1
            )
        )


    # =====================================

    def encode(
        self,
        x,
        edge_index
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = self.bn1(x)

        x = F.relu(x)

        x = self.dropout(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = self.bn2(x)

        return x


    # =====================================

    def decode(
        self,
        z,
        edge_label_index
    ):

        src = edge_label_index[0]

        dst = edge_label_index[1]

        edge_embeddings = torch.cat(
            [
                z[src],
                z[dst]
            ],
            dim=1
        )

        return self.decoder(
            edge_embeddings
        ).squeeze()


# ====================================================
# DEVICE
# ====================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "\nUsing device:",
    device
)

model = GraphSAGELinkPredictor(
    input_dim
).to(device)

train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

criterion = nn.BCEWithLogitsLoss()

from sklearn.metrics import roc_auc_score
import copy

# ====================================================
# TRAIN FUNCTION
# ====================================================

def train():

    model.train()

    optimizer.zero_grad()

    z = model.encode(
        train_data.x,
        train_data.edge_index
    )

    logits = model.decode(
        z,
        train_data.edge_label_index
    )

    loss = criterion(
        logits,
        train_data.edge_label.float()
    )

    loss.backward()

    optimizer.step()

    return loss.item()


# ====================================================
# VALIDATION FUNCTION
# ====================================================

@torch.no_grad()
def validate():

    model.eval()

    z = model.encode(
        val_data.x,
        val_data.edge_index
    )

    logits = model.decode(
        z,
        val_data.edge_label_index
    )

    probs = torch.sigmoid(
        logits
    ).cpu().numpy()

    y_true = (
        val_data.edge_label
        .cpu()
        .numpy()
    )

    auc = roc_auc_score(
        y_true,
        probs
    )

    return auc


# ====================================================
# TRAIN LOOP
# ====================================================

NUM_EPOCHS = 500

best_auc = 0

best_model = None

patience = 50

counter = 0

history = []

print("\nTraining...\n")

for epoch in range(1, NUM_EPOCHS + 1):

    loss = train()

    val_auc = validate()

    history.append(
        (
            epoch,
            loss,
            val_auc
        )
    )

    if val_auc > best_auc:

        best_auc = val_auc

        best_model = copy.deepcopy(
            model.state_dict()
        )

        counter = 0

    else:

        counter += 1

    if epoch % 20 == 0:

        print(
            f"Epoch {epoch:03d} | "
            f"Loss {loss:.4f} | "
            f"Val AUC {val_auc:.4f}"
        )

    # ====================================
    # EARLY STOPPING
    # ====================================

    if counter >= patience:

        print(
            "\nEarly stopping triggered."
        )

        break


# ====================================================
# LOAD BEST MODEL
# ====================================================

print(
    "\nBest Validation AUC:",
    round(best_auc, 4)
)

model.load_state_dict(
    best_model
)


# ====================================================
# SAVE MODEL
# ====================================================

torch.save(
    model.state_dict(),
    "models/graphsage_link_predictor.pt"
)

print(
    "Saved model:"
)

print(
    "models/graphsage_link_predictor.pt"
)
torch.save(
    model.state_dict(),
    "models/graphsage_link_predictor.pt"
)
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ====================================================
# TEST FUNCTION
# ====================================================

@torch.no_grad()
def evaluate():

    model.eval()

    z = model.encode(
        test_data.x,
        test_data.edge_index
    )

    logits = model.decode(
        z,
        test_data.edge_label_index
    )

    probs = torch.sigmoid(
        logits
    ).cpu().numpy()

    y_true = (
        test_data.edge_label
        .cpu()
        .numpy()
    )

    y_pred = (
        probs > 0.5
    ).astype(int)

    acc = accuracy_score(
        y_true,
        y_pred
    )

    auc = roc_auc_score(
        y_true,
        probs
    )

    ap = average_precision_score(
        y_true,
        probs
    )

    f1 = f1_score(
        y_true,
        y_pred
    )

    return (
        y_true,
        y_pred,
        probs,
        acc,
        auc,
        ap,
        f1
    )


# ====================================================
# FINAL EVALUATION
# ====================================================

(
    y_true,
    y_pred,
    probs,
    acc,
    auc,
    ap,
    f1
) = evaluate()


print("\n")
print("=" * 50)
print("FINAL TEST RESULTS")
print("=" * 50)

print(
    "Accuracy:",
    round(acc,4)
)

print(
    "ROC-AUC:",
    round(auc,4)
)

print(
    "Average Precision:",
    round(ap,4)
)

print(
    "F1 Score:",
    round(f1,4)
)


# ====================================================
# CONFUSION MATRIX
# ====================================================

print("\nConfusion Matrix\n")

cm = confusion_matrix(
    y_true,
    y_pred
)

print(cm)


# ====================================================
# CLASSIFICATION REPORT
# ====================================================

print("\nClassification Report\n")

print(
    classification_report(
        y_true,
        y_pred
    )
)


# ====================================================
# SAMPLE PREDICTIONS
# ====================================================

print("\nSample Predictions\n")

edge_index = (
    test_data.edge_label_index
    .cpu()
)

for i in range(
    min(30, len(probs))
):

    src = (
        edge_index[0][i]
        .item()
    )

    dst = (
        edge_index[1][i]
        .item()
    )

    label = int(
        y_true[i]
    )

    prediction = int(
        y_pred[i]
    )

    score = probs[i]

    print(
        f"({src},{dst}) "
        f"True={label} "
        f"Pred={prediction} "
        f"Prob={score:.3f}"
    )


# ====================================================
# TOP PREDICTED LINKS
# ====================================================

print("\nTop Predicted Links\n")

pairs = []

for i in range(
    len(probs)
):

    src = (
        edge_index[0][i]
        .item()
    )

    dst = (
        edge_index[1][i]
        .item()
    )

    pairs.append(
        (
            src,
            dst,
            probs[i]
        )
    )

pairs = sorted(
    pairs,
    key=lambda x: x[2],
    reverse=True
)

for src, dst, score in pairs[:20]:

    print(
        f"({src},{dst}) -> "
        f"{score:.4f}"
    )


print("\n")
print("=" * 60)
print("LINK PREDICTION PIPELINE COMPLETE")
print("=" * 60)