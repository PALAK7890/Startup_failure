import pandas as pd
import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GCNConv

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score
)


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

X = torch.tensor(
    features_df.drop(
        columns=["topic_id"]
    ).values,
    dtype=torch.float
)

input_dim = X.shape[1]

print("Feature shape:", X.shape)


# ====================================================
# LOAD EDGES
# ====================================================

print("Loading graph edges...")

edges_df = pd.read_csv(
    "data/graph/edges.csv"
)

edge_index = torch.tensor(
    [
        edges_df["source"].values,
        edges_df["target"].values
    ],
    dtype=torch.long
)

# make graph undirected
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
# TRAIN / VAL / TEST SPLIT
# ====================================================

transform = RandomLinkSplit(
    num_val=0.1,
    num_test=0.2,
    is_undirected=True,
    add_negative_train_samples=True,
    neg_sampling_ratio=1.0
)

train_data, val_data, test_data = transform(data)

print()
print("Train edges:", train_data.edge_label.shape[0])
print("Val edges:", val_data.edge_label.shape[0])
print("Test edges:", test_data.edge_label.shape[0])


# ====================================================
# MODEL
# ====================================================

class LinkPredictor(torch.nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.conv1 = GCNConv(
            input_dim,
            128
        )

        self.conv2 = GCNConv(
            128,
            64
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
        edge_label_index
    ):

        src = edge_label_index[0]
        dst = edge_label_index[1]

        return (
            z[src] * z[dst]
        ).sum(dim=1)


# ====================================================
# INITIALIZE MODEL
# ====================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nUsing device:", device)

model = LinkPredictor(
    input_dim
).to(device)

train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

criterion = torch.nn.BCEWithLogitsLoss()


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
# EVALUATION
# ====================================================

@torch.no_grad()
def evaluate(data_split):

    model.eval()

    z = model.encode(
        data_split.x,
        data_split.edge_index
    )

    logits = model.decode(
        z,
        data_split.edge_label_index
    )

    probs = torch.sigmoid(
        logits
    ).cpu().numpy()

    y_true = (
        data_split.edge_label
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

    return acc, auc, ap, f1


# ====================================================
# TRAINING LOOP
# ====================================================

print("\nTraining...\n")

for epoch in range(1, 201):

    loss = train()

    if epoch % 20 == 0:

        val_acc, val_auc, val_ap, val_f1 = evaluate(
            val_data
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Loss {loss:.4f} | "
            f"Val AUC {val_auc:.4f}"
        )


# ====================================================
# FINAL TEST
# ====================================================

test_acc, test_auc, test_ap, test_f1 = evaluate(
    test_data
)

print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")

print(
    "Accuracy:",
    round(test_acc,4)
)

print(
    "ROC-AUC:",
    round(test_auc,4)
)

print(
    "Average Precision:",
    round(test_ap,4)
)

print(
    "F1 Score:",
    round(test_f1,4)
)


# ====================================================
# SAMPLE PREDICTIONS
# ====================================================

print("\nSample Predictions\n")

with torch.no_grad():

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
    )

for i in range(30):

    src = (
        test_data.edge_label_index[0][i]
        .item()
    )

    dst = (
        test_data.edge_label_index[1][i]
        .item()
    )

    label = (
        test_data.edge_label[i]
        .item()
    )

    score = (
        probs[i]
        .item()
    )

    print(
        f"({src},{dst}) "
        f"True={label} "
        f"Prob={score:.3f}"
    )