from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import os

# =====================================================
# LOAD DATA
# =====================================================

print("Loading papers...")

df = pd.read_csv(
    "data/processed/clean_papers.csv"
)

df = df.dropna(
    subset=["abstract"]
)

documents = (
    df["abstract"]
    .astype(str)
    .tolist()
)

print(
    f"Documents loaded: {len(documents)}"
)

# =====================================================
# LOAD MODEL
# =====================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# =====================================================
# GENERATE EMBEDDINGS
# =====================================================

print("\nGenerating embeddings...")

embeddings = model.encode(
    documents,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print(
    "\nEmbedding shape:",
    embeddings.shape
)

# =====================================================
# SAVE EMBEDDINGS
# =====================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

np.save(
    "data/processed/paper_embeddings.npy",
    embeddings
)

print(
    "Saved paper_embeddings.npy"
)

# =====================================================
# OPTIONAL CSV VERSION
# =====================================================

embedding_df = pd.DataFrame(
    embeddings
)

embedding_df.insert(
    0,
    "paper_id",
    df.index
)

embedding_df.to_csv(
    "data/processed/paper_embeddings.csv",
    index=False
)

print(
    "Saved paper_embeddings.csv"
)

# =====================================================
# SUMMARY
# =====================================================

print("\nDone!")

print(
    "Shape:",
    embeddings.shape
)