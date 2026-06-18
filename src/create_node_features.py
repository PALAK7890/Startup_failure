import pickle
import numpy as np
import pandas as pd

print("Loading model...")

with open(
    "models/bertopic_model.pkl",
    "rb"
) as f:
    topic_model = pickle.load(f)

topic_info = pd.read_csv(
    "data/processed/topic_info.csv"
)

topic_info = topic_info[
    topic_info["Topic"] != -1
]

topic_embeddings = topic_model.topic_embeddings_

features = []

for idx, row in topic_info.iterrows():

    topic_id = int(row["Topic"])

    embedding = topic_embeddings[topic_id]

    feature_row = {
        "topic_id": topic_id
    }

    for i, value in enumerate(embedding):

        feature_row[f"f_{i}"] = value

    features.append(feature_row)

features_df = pd.DataFrame(features)

features_df.to_csv(
    "data/graph/node_features.csv",
    index=False
)

print(features_df.shape)
print("Saved node_features.csv")
