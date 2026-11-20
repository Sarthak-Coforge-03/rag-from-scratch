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
| `EMBED_PROVIDER` | `sentence-transformers` | embedding library (`--create` only) |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | embedding model (`--create` only) |
| `EMBED_BATCH_SIZE` | `64` | chunks embedded per batch |

## Indexes

An index is a folder under `INDEX_ROOT` holding the chunked documents (SQLite) and
their vectors (one folder per embedding model). Its chunker settings are fixed when it is created.

### Create a new index

Uses the `CHUNK_*` settings. The index is named `<strategy>-<size>`; if that name is
taken, `_1`, `_2`, … is added, so an existing index is never overwritten.

```sh
CHUNK_STRATEGY=structural CHUNK_SIZE=800 CHUNK_OVERLAP=0 uv run python scripts/02_chunk.py
# -> Creating new index 'structural-800' ... 13 added
```

### Update an index after the documents change

Pass the index name. It reuses the settings the index was created with (`CHUNK_*` is
ignored) and only processes files that were added, edited or deleted.

```sh
uv run python scripts/02_chunk.py --sync structural-800
# -> Documents : 0 added, 1 updated, 12 unchanged, 0 removed
```

An unknown name stops with `Error: Index 'NAME' not found` and exit code 1.

### Embed an index

Run after the chunk step. `--create` builds the vector folder for `EMBED_MODEL`; if
that model already has one, it is deleted and every chunk is embedded again.

```sh
uv run python scripts/03_embed.py --create structural-800
# -> Chunks : 73 embedded, 0 deleted, 0 unchanged (73 in SQL)
```

### Sync vectors after the documents change

The chunk step only updates SQL, so after `02_chunk.py --sync` the vector folders are
out of date. `--sync` updates **every** vector folder in the index, each with the model
it was created with (`EMBED_MODEL` is ignored): new chunks are embedded, removed ones
deleted, the rest left alone.

```sh
uv run python scripts/02_chunk.py --sync structural-800    # 1. documents -> SQL
uv run python scripts/03_embed.py --sync structural-800    # 2. SQL -> every vector folder
```

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
  chunker_config.json    the same settings, read back on --sync
  sql/docs.db            documents and chunk text
  vectors/<model>/       one folder per embedding model
    embedder.txt         SentenceTransformerEmbedder(all-MiniLM-L6-v2)
    embedder_config.json the same model, read back on --sync
```

To compare chunking settings, create one index per setting (e.g. `structural-800`
and `token-800`) and point later steps at the one you want.

## Ask questions

Needs a chunked and embedded index, plus `LLM_ENDPOINT` (ending in `/anthropic`)
and `ANTHROPIC_FOUNDRY_API_KEY` in `.env`.

```sh
uv run rag-from-scratch
```

Menus pick the index, retriever (vector / keyword / hybrid), whether to rerank, and
the LLM; your `.env` values are preselected. Then type questions; an empty question
(or Ctrl-C) quits. Each answer is followed by every chunk it was shown, with the
ones it cited marked `*`:

```
Answer:
Expenses over $2,500 need director approval.

Sources (* = cited):
 * [1] expense_reimbursement_policy                 EXP-001 v3.0   score   4.952
   [2] travel_policy                                TRV-001        score  -2.310
```

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
