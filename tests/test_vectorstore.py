"""Tests for the Chroma vector store: ranking, overwriting and deleting by chunk id.

Uses tiny 3-number vectors instead of a real embedding model, so the expected
ranking is obvious and the tests run fast. Cosine distance compares directions,
so each test vector points along a different axis.
"""

from rag.vectorstore import ChromaVectorStore

POINTS_ALONG_X = [1.0, 0.0, 0.0]
POINTS_ALONG_Y = [0.0, 1.0, 0.0]
POINTS_ALONG_Z = [0.0, 0.0, 1.0]
MOSTLY_X_A_LITTLE_Y = [0.9, 0.1, 0.0]


def make_store(tmp_path):
    """Return an empty store in this test's own temporary folder, isolated from other tests."""
    return ChromaVectorStore(persist_directory=tmp_path, collection_name="test_chunks")


def test_query_returns_nearest_chunks_first(tmp_path):
    """Hits come back closest first, with non-decreasing distance, and at most top_k of them."""
    store = make_store(tmp_path)
    store.upsert(
        chunk_ids=["x_chunk", "y_chunk", "z_chunk"],
        embeddings=[POINTS_ALONG_X, POINTS_ALONG_Y, POINTS_ALONG_Z],
    )

    matches = store.query(embedding=MOSTLY_X_A_LITTLE_Y, top_k=2)

    assert [match.chunk_id for match in matches] == ["x_chunk", "y_chunk"]
    assert matches[0].distance <= matches[1].distance


def test_upsert_overwrites_and_delete_removes(tmp_path):
    """Writing the same ids again replaces their vectors (no duplicates); delete removes an id."""
    store = make_store(tmp_path)
    store.upsert(
        chunk_ids=["x_chunk", "y_chunk", "z_chunk"],
        embeddings=[POINTS_ALONG_X, POINTS_ALONG_Y, POINTS_ALONG_Z],
    )

    # Swap the vectors of x_chunk and z_chunk by upserting the same ids again.
    store.upsert(
        chunk_ids=["x_chunk", "z_chunk"],
        embeddings=[POINTS_ALONG_Z, POINTS_ALONG_X],
    )
    assert store.count() == 3
    nearest_after_overwrite = store.query(embedding=MOSTLY_X_A_LITTLE_Y, top_k=1)
    assert nearest_after_overwrite[0].chunk_id == "z_chunk"

    store.delete(chunk_ids=["z_chunk"])
    assert store.count() == 2
    remaining_ids = [
        match.chunk_id for match in store.query(MOSTLY_X_A_LITTLE_Y, top_k=3)
    ]
    assert "z_chunk" not in remaining_ids
