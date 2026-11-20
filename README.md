# rag-from-scratch

Q&A over company policy documents (`data/policies/`) using chunking, embeddings and a
vector store.

## Setup

```sh
uv sync                  # install Python 3.12 dependencies into .venv
cp .env.example .env     # optional: override settings (defaults are in src/rag/config.py)
```

Settings (environment variables or `.env`):

| Variable | Default | Used for |
|---|---|---|
| `POLICY_DOCS_DIR` | `data/policies` | folder of markdown files to ingest |
| `MAX_DOC_BYTES` | `5000000` | larger files are skipped with a warning |
| `INDEX_ROOT` | `indexes` | where index folders are created |
| `CHUNK_STRATEGY` | `token` | `token`, `recursive` or `structural` |
| `CHUNK_SIZE` | `800` | characters per chunk |
| `CHUNK_OVERLAP` | `150` | characters; ignored (with a warning) by `recursive` and `structural` |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |

## Indexes

An index is a folder under `INDEX_ROOT` holding the chunked documents (SQLite) and, later,
their vectors. Its chunker settings are fixed when it is created.

### Create a new index

Uses the `CHUNK_*` settings. The index is named `<strategy>-<size>`; if that name is
taken, `_1`, `_2`, … is added, so an existing index is never overwritten.

```sh
CHUNK_STRATEGY=structural CHUNK_SIZE=800 CHUNK_OVERLAP=0 uv run python scripts/02_chunk.py
# -> Creating new index 'structural-800' ... added 13
```

### Update an index after the documents change

Pass the index name. It reuses the settings the index was created with (`CHUNK_*` is
ignored) and only processes files that were added, edited or deleted.

```sh
uv run python scripts/02_chunk.py --index structural-800
# -> added 0 / updated 1 / unchanged 12 / removed 0
```

An unknown name stops with `Error: Index 'NAME' not found` and exit code 1.

### See or delete indexes

```sh
ls indexes/                                   # all indexes
cat indexes/structural-800/chunker.txt        # how one was chunked
rm -rf indexes/structural-800                 # delete one (then create it again)
```

What an index folder holds:

```
indexes/structural-800/
  chunker.txt            StructuralChunker(size=800, overlap=0)
  chunker_config.json    the same settings, read back on --index updates
  sql/docs.db            documents and chunk text
  vectors/<model>/       one folder per embedding model (embed step, coming next)
```

To compare chunking settings, create one index per setting (e.g. `structural-800`
and `token-800`) and point later steps at the one you want.

## Other commands

```sh
uv run python scripts/01_minimal_loop.py     # embed two texts, query, print which ranks first
uv run pytest                                # all tests
uv run pytest tests/test_ingest.py::test_edited_file_is_rechunked_and_deleted_file_is_removed
uv run ruff format .                         # format
```

## Note on the corpus

`data/policies/expense_reimbursement_policy_2023.md` is a deliberately outdated duplicate
of `expense_reimbursement_policy.md`. Leave it in place; see
[docs/data-quality-issue.md](docs/data-quality-issue.md).
