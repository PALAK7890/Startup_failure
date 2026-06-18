from sentence_transformers import SentenceTransformer
import pandas as pd

df = pd.read_csv(
    "data/processed/clean_papers.csv"
)

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

embeddings = model.encode(
    df["abstract"].tolist(),
    show_progress_bar=True
)

print(embeddings.shape)