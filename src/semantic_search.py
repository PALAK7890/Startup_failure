import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ==================================================
# LOAD DATA
# ==================================================

print("Loading papers...")

papers_df = pd.read_csv(
    "data/processed/clean_papers.csv"
)

embeddings = np.load(
    "data/processed/paper_embeddings.npy"
)

print(
    "Embedding shape:",
    embeddings.shape
)


# ==================================================
# LOAD MODEL
# ==================================================

print("Loading model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ==================================================
# QUERY
# ==================================================

query = input(
    "\nEnter search query: "
)


# ==================================================
# QUERY EMBEDDING
# ==================================================

query_embedding = model.encode(
    [query],
    normalize_embeddings=True
)


# ==================================================
# COSINE SIMILARITY
# ==================================================

scores = cosine_similarity(
    query_embedding,
    embeddings
)[0]


# ==================================================
# TOP RESULTS
# ==================================================

TOP_K = 10

top_indices = np.argsort(
    scores
)[::-1][:TOP_K]


print("\nTop Papers\n")

for rank, idx in enumerate(
    top_indices,
    start=1
):

    print("=" * 80)

    print(
        f"Rank {rank}"
    )

    if "title" in papers_df.columns:

        print(
            "Title:",
            papers_df.iloc[idx]["title"]
        )

    print()

    print(
        "Similarity:",
        round(
            float(scores[idx]),
            4
        )
    )

    abstract = str(
        papers_df.iloc[idx]["abstract"]
    )

    print()

    print(
        abstract[:500]
    )

    print()