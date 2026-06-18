import arxiv
import pandas as pd
from tqdm import tqdm

client = arxiv.Client()

queries = [
    "cs.AI",
    "cs.LG",
    "cs.CL",
    "cs.CV",
    "cs.RO",
    "cs.IR"
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