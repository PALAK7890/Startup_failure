import pickle
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# LOAD MODEL
# =====================================================

print("\nLoading BERTopic model...")

with open(
    "models/bertopic_model.pkl",
    "rb"
) as f:
    topic_model = pickle.load(f)

# =====================================================
# LOAD TOPIC INFO
# =====================================================

topic_info = pd.read_csv(
    "data/processed/topic_info.csv"
)

# Remove outlier topic (-1)
topic_info = topic_info[
    topic_info["Topic"] != -1
].reset_index(drop=True)

# =====================================================
# GET TOPIC EMBEDDINGS
# =====================================================

topic_embeddings = topic_model.topic_embeddings_

print(
    f"\nTopic Embeddings Shape: "
    f"{topic_embeddings.shape}"
)

# =====================================================
# COSINE SIMILARITY
# =====================================================

similarity_matrix = cosine_similarity(
    topic_embeddings
)

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def get_topic_keywords(topic_id, top_n=5):

    words = topic_model.get_topic(topic_id)

    if words is None:
        return "Unknown"

    return ", ".join(
        [word[0] for word in words[:top_n]]
    )

def get_topic_size(topic_id):

    row = topic_info[
        topic_info["Topic"] == topic_id
    ]

    if len(row) == 0:
        return 0

    return int(row["Count"].values[0])

# =====================================================
# GAP DETECTION
# =====================================================

research_gaps = []

num_topics = len(topic_embeddings)

for i in range(num_topics):

    for j in range(i + 1, num_topics):

        similarity = similarity_matrix[i, j]

        topic_a_size = get_topic_size(i)
        topic_b_size = get_topic_size(j)

        # Approximate co-occurrence penalty
        co_occurrence_penalty = (
            min(topic_a_size, topic_b_size) /
            max(topic_a_size, topic_b_size)
        )

        gap_score = (
            similarity *
            (1 - co_occurrence_penalty)
        )

        research_gaps.append({
            "topic_a": i,
            "topic_b": j,
            "similarity": round(similarity, 4),
            "gap_score": round(gap_score, 4),
            "topic_a_keywords":
                get_topic_keywords(i),
            "topic_b_keywords":
                get_topic_keywords(j)
        })

# =====================================================
# SORT
# =====================================================

research_gaps = sorted(
    research_gaps,
    key=lambda x: x["gap_score"],
    reverse=True
)

# =====================================================
# SAVE RESULTS
# =====================================================

results_df = pd.DataFrame(
    research_gaps
)

results_df.to_csv(
    "outputs/research_gaps.csv",
    index=False
)

# =====================================================
# DISPLAY TOP OPPORTUNITIES
# =====================================================

print("\n")
print("=" * 80)
print("TOP RESEARCH OPPORTUNITIES")
print("=" * 80)

for idx, gap in enumerate(
    research_gaps[:20],
    start=1
):

    print("\n")

    print(f"Opportunity #{idx}")

    print(
        f"Gap Score: "
        f"{gap['gap_score']}"
    )

    print(
        f"Semantic Similarity: "
        f"{gap['similarity']}"
    )

    print(
        f"\nTopic A:\n"
        f"{gap['topic_a_keywords']}"
    )

    print(
        f"\nTopic B:\n"
        f"{gap['topic_b_keywords']}"
    )

    print("-" * 80)

print(
    "\nSaved to outputs/research_gaps.csv"
)