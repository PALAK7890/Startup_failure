import pandas as pd
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity

# ============================================
# LOAD MODEL
# ============================================

print("Loading BERTopic model...")

topic_model = BERTopic.load(
    "models/bertopic_model"
)

# ============================================
# LOAD TOPIC LABELS
# ============================================

labels_df = pd.read_csv(
    "data/graph/topic_labels.csv"
)

topic_to_label = dict(
    zip(
        labels_df["topic_id"],
        labels_df["label"]
    )
)

# ============================================
# QUERY
# ============================================

query = input(
    "\nEnter topic query: "
)

# ============================================
# FIND CLOSEST TOPIC
# ============================================

topics, similarities = topic_model.find_topics(
    query,
    top_n=1
)

seed_topic = topics[0]

print("\nMatched Topic")

print(
    seed_topic,
    topic_to_label.get(
        seed_topic,
        ""
    )
)

# ============================================
# TOPIC EMBEDDINGS
# ============================================

topic_embeddings = (
    topic_model.topic_embeddings_
)

query_embedding = topic_embeddings[
    seed_topic
]

# ============================================
# COSINE SIMILARITIES
# ============================================

scores = cosine_similarity(
    [query_embedding],
    topic_embeddings
)[0]

pairs = []

for topic_id, score in enumerate(scores):

    if topic_id == seed_topic:
        continue

    pairs.append(
        (
            topic_id,
            score
        )
    )

pairs = sorted(
    pairs,
    key=lambda x:x[1],
    reverse=True
)

# ============================================
# RECOMMENDATIONS
# ============================================

print("\nTop Related Topics\n")

for topic_id, score in pairs[:10]:

    words = topic_model.get_topic(
        topic_id
    )

    if words is None:
        continue

    keywords = ", ".join(
        [
            x[0]
            for x in words[:5]
        ]
    )

    print(
        f"{topic_id:3d}",
        f"{score:.3f}",
        keywords
    )