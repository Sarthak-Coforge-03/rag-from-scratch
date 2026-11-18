"""Minimal embed -> store -> retrieve loop on two known pieces of text.

Lab "Suggested Approach" steps 2-3: prove that sentence-transformers embeddings can be
stored in a chromadb collection and that a query returns the more relevant of two texts,
before any chunking, hybrid search or reranking is layered on top.

Run with:  uv run python scripts/01_minimal_loop.py
"""

import chromadb
from sentence_transformers import SentenceTransformer

from rag.config import settings

MEAL_TEXT = "Employees may claim meals up to $75 per day when traveling overnight."
PASSWORD_TEXT = "Passwords must be at least 14 characters long."
QUESTION = "How much can I spend on food per day on a business trip?"


def main() -> None:
    """Embed one text, store it, add a second, then query and print the ranking."""
    print(f"Loading embedding model: {settings.embed_model}")
    model = SentenceTransformer(settings.embed_model)

    # Step 1: embed one short piece of text.
    meal_embedding = model.encode(MEAL_TEXT).tolist()
    print(f"\n[1] Embedded text A -> vector of length {len(meal_embedding)}")
    print(f"    text A: {MEAL_TEXT!r}")

    # Step 2: store it in an in-memory chromadb collection, using OUR embedding
    # (embedding_function=None stops Chroma from using its own built-in model).
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name="minimal_loop",
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )
    collection.add(ids=["text_a"], documents=[MEAL_TEXT], embeddings=[meal_embedding])
    print(
        f"\n[2] Stored text A in collection 'minimal_loop' -> count = {collection.count()}"
    )

    # Step 3: add a second, unrelated piece of text.
    password_embedding = model.encode(PASSWORD_TEXT).tolist()
    collection.add(
        ids=["text_b"], documents=[PASSWORD_TEXT], embeddings=[password_embedding]
    )
    print(f"\n[3] Stored text B -> count = {collection.count()}")
    print(f"    text B: {PASSWORD_TEXT!r}")

    # Step 4: embed a question and ask Chroma for the closest texts.
    question_embedding = model.encode(QUESTION).tolist()
    results = collection.query(query_embeddings=[question_embedding], n_results=2)

    print(f"\n[4] Query: {QUESTION!r}")
    print("    Results, most similar first (cosine distance: lower = more similar):")
    result_ids = results["ids"][0]
    result_texts = results["documents"][0]
    result_distances = results["distances"][0]
    for rank in range(len(result_ids)):
        print(
            f"    #{rank + 1}  {result_ids[rank]}  "
            f"distance={result_distances[rank]:.4f}  {result_texts[rank]!r}"
        )


if __name__ == "__main__":
    main()
