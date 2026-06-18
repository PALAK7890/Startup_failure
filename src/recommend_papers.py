import pandas as pd
from bertopic import BERTopic

# ======================================
# LOAD MODEL
# ======================================

print("Loading BERTopic model...")

topic_model = BERTopic.load(
    "models/bertopic_model"
)

# ======================================
# LOAD PAPERS
# ======================================

papers_df = pd.read_csv(
    "data/processed/papers_with_topics.csv"
)

# ======================================
# USER QUERY
# ======================================

query = input(
    "\nEnter query: "
)

# ======================================
# FIND TOPICS
# ======================================

topics, similarities = topic_model.find_topics(
    query,
    top_n=5
)

print("\nRelevant Topics")

for t, s in zip(topics, similarities):

    words = topic_model.get_topic(t)

    if words is None:
        continue

    label = ", ".join(
        [
            x[0]
            for x in words[:5]
        ]
    )

    print(
        f"Topic {t}: {label} ({s:.3f})"
    )

# ======================================
# RETRIEVE PAPERS
# ======================================

results = papers_df[
    papers_df["topic"].isin(topics)
]

print(
    "\nRetrieved Papers:",
    len(results)
)

# ======================================
# DISPLAY PAPERS
# ======================================

top_k = 10

print("\nTop Papers\n")

for idx, row in results.head(top_k).iterrows():

    print("=" * 80)

    if "title" in row:
        print(
            "Title:",
            row["title"]
        )

    print()

    print(
        "Topic:",
        row["topic"]
    )

    abstract = str(
        row["abstract"]
    )

    print(
        abstract[:500]
    )

    print()