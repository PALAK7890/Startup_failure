import pandas as pd

df = pd.read_csv("arxiv_papers.csv")

df = df.drop_duplicates(subset=["title"])
df = df.dropna(subset=["abstract"])

df["abstract"] = df["abstract"].str.replace("\n", " ")

df.to_csv(
    "data/processed/clean_papers.csv",
    index=False
)

print(df.shape)