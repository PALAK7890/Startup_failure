import arxiv
import pandas as pd
from tqdm import tqdm

client = arxiv.Client()

queries = [
    "cat:cs.CL",   # NLP / LLMs
    "cat:cs.CV",   # Computer Vision
    "cat:cs.RO",   # Robotics
    "cat:cs.LG",   # Machine Learning
    "cat:cs.AI",   # AI
    "cat:cs.IR",   # Information Retrieval
    "cat:q-bio.QM",# Medical Imaging/Bio
    "cat:stat.ML"
]

papers = []

for query in queries:

    print(f"\nCollecting papers for: {query}")

    search = arxiv.Search(
        query=query,
        max_results=500,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    for result in tqdm(client.results(search)):

        papers.append({
            "query": query,
            "title": result.title,
            "abstract": result.summary,
            "published": result.published,
            "authors": ", ".join(
                [a.name for a in result.authors]
            ),
            "categories": ",".join(
                result.categories
            )
        })

df = pd.DataFrame(papers)

print("\nBefore deduplication:")
print(df.shape)

df = df.drop_duplicates(
    subset=["title"]
)

print("\nAfter deduplication:")
print(df.shape)

df.to_csv(
    "arxiv_papers.csv",
    index=False
)

print(
    f"\nCollected {len(df)} unique papers"
)