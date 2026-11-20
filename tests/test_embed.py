"""Tests for the embed step: keeping a vector folder in sync with the chunks in SQL."""

from rag.chunking import create_chunker
from rag.documents import DocumentRepository
from rag.embed import sync_vectors
from rag.embedding import BaseEmbedder
from rag.ingest import Ingestor
from rag.vectorstore import ChromaVectorStore

NO_SIZE_LIMIT = 5_000_000


class CountingEmbedder(BaseEmbedder):
    """Fake embedder: a 2-number vector per text (no model download) that counts its work."""

    def __init__(self):
        """Batch size 2, so syncing a few chunks already needs several batches."""
        super().__init__(model_name="fake", batch_size=2)
        self.texts_embedded = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return [length, 1.0] for each text and count how many texts were embedded."""
        self.texts_embedded += len(texts)
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Not used by these tests."""
        raise NotImplementedError


def make_ingestor(docs_folder):
    """Write two policy files (several small chunks each) and ingest them into in-memory SQL."""
    (docs_folder / "pto.md").write_text("# PTO\n\n" + "Paid time off rules. " * 20)
    (docs_folder / "travel.md").write_text("# Travel\n\n" + "Book flights early. " * 20)
    repository = DocumentRepository("sqlite://")
    chunker = create_chunker(strategy="token", chunk_size=100, chunk_overlap=0)
    ingestor = Ingestor(repository=repository, chunker=chunker)
    ingestor.ingest_folder(docs_folder, NO_SIZE_LIMIT)
    return ingestor


def test_second_sync_embeds_nothing(tmp_path):
    """The first sync embeds every chunk in SQL; syncing again finds nothing to do."""
    docs_folder = tmp_path / "docs"
    docs_folder.mkdir()
    ingestor = make_ingestor(docs_folder)
    store = ChromaVectorStore(location=tmp_path / "vectors", collection_name="chunks")
    embedder = CountingEmbedder()
    chunks_in_sql = len(ingestor.repository.get_all_chunk_ids())

    first_report = sync_vectors(ingestor.repository, embedder, store)
    second_report = sync_vectors(ingestor.repository, embedder, store)

    assert first_report.embedded == chunks_in_sql
    assert second_report.embedded == 0
    assert second_report.deleted == 0
    assert second_report.unchanged == chunks_in_sql
    assert embedder.texts_embedded == chunks_in_sql  # the model was not called again


def test_sync_after_rechunk_embeds_only_new_chunks_and_deletes_old_ones(tmp_path):
    """Edit one doc and delete the other: only the edited doc is embedded again.

    Afterwards the vector folder holds exactly the chunk ids that are in SQL.
    """
    docs_folder = tmp_path / "docs"
    docs_folder.mkdir()
    ingestor = make_ingestor(docs_folder)
    store = ChromaVectorStore(location=tmp_path / "vectors", collection_name="chunks")
    sync_vectors(ingestor.repository, CountingEmbedder(), store)
    ids_before = store.list_ids()

    (docs_folder / "pto.md").write_text("# PTO\n\n" + "New paid time off rules. " * 20)
    (docs_folder / "travel.md").unlink()
    ingestor.ingest_folder(docs_folder, NO_SIZE_LIMIT)
    embedder = CountingEmbedder()
    report = sync_vectors(ingestor.repository, embedder, store)

    new_pto_ids = ingestor.repository.get_all_chunk_ids()
    assert report.embedded == len(new_pto_ids)
    assert report.deleted == len(ids_before)  # every old id changed or was removed
    assert report.unchanged == 0
    assert embedder.texts_embedded == len(new_pto_ids)
    assert store.list_ids() == new_pto_ids
