import pandas as pd
import torch
from bertopic import BERTopic

from link_prediction import GraphSAGELinkPredictor

# =====================================
# LOAD TOPIC LABELS
# =====================================

labels_df = pd.read_csv(
    "data/graph/topic_labels.csv"
)

nodes_df = pd.read_csv(
    "data/graph/nodes.csv"
)

features_df = pd.read_csv(
    "data/graph/node_features.csv"
)

# =====================================
# LOAD MODEL
# =====================================

topic_model = BERTopic.load(
    "models/bertopic_model"
)

# =====================================
# LOAD GNN
# =====================================

input_dim = (
    features_df.shape[1] - 1
)

model = GraphSAGELinkPredictor(
    input_dim
)

model.load_state_dict(
    torch.load(
        "models/graphsage_link_predictor.pt",
        map_location="cpu"
    )
)

model.eval()
query = "agentic rag"

topic_id, prob = topic_model.find_topics(
    query,
    top_n=1
)

seed_topic = topic_id[0]

print(
    "Matched Topic:",
    seed_topic
)
topic_embeddings = (
    topic_model.topic_embeddings_
)

query_embedding = (
    topic_embeddings[seed_topic]
)

from sklearn.metrics.pairwise import cosine_similarity

scores = cosine_similarity(
    [query_embedding],
    topic_embeddings
)[0]

pairs = []

for i, score in enumerate(scores):

    if i == seed_topic:
        continue

    pairs.append(
        (
            i,
            score
        )
    )

pairs = sorted(
    pairs,
    key=lambda x:x[1],
    reverse=True
)
print("\nRecommended Topics\n")

for topic, score in pairs[:10]:

    words = topic_model.get_topic(
        topic
    )

    label = ", ".join(
        [
            x[0]
            for x in words[:5]
        ]
    )

    print(
        label,
        round(score,3)
    )