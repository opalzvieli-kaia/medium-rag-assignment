# Submission Checklist

## Before Full Ingestion

- Replace `LLMOD_API_KEY` with a valid key that starts with `sk-`.
- Replace `PINECONE_API_KEY` with a valid Pinecone API key.
- Confirm `PINECONE_INDEX_NAME` is set.
- Run:

```bash
python3 scripts/check_env.py
```

## Cheap Test

Run a subset ingestion first:

```bash
python3 scripts/ingest.py --limit 100
```

Run the assignment-style test questions:

```bash
python3 scripts/test_queries.py
```

## Full Ingestion

Only after the subset test works:

```bash
python3 scripts/ingest.py
```

Current status: completed from a clean namespace reset. Pinecone namespace `medium-articles` contains `22,174` vectors.

## Vercel

Set these environment variables in Vercel:

- `LLMOD_API_KEY`
- `LLMOD_BASE_URL`
- `LLMOD_CHAT_MODEL`
- `LLMOD_EMBEDDING_MODEL`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `PINECONE_NAMESPACE`
- `RAG_CHUNK_SIZE`
- `RAG_OVERLAP_RATIO`
- `RAG_TOP_K`
- `EMBEDDING_DIMENSIONS`

## Required Submission

- Public live URL
- Public GitHub URL

Keep the Pinecone index active until the grade is received.
