# Medium Article RAG Assistant

Public API for the Individual RAG Assignment. The assistant answers questions strictly from the provided Medium articles CSV using Retrieval-Augmented Generation.

## Configuration

Chosen RAG parameters:

- `chunk_size`: `512`
- `overlap_ratio`: `0.2`
- `top_k`: `7`
- embedding model: `4UHRUIN-text-embedding-3-small`
- chat model: `4UHRUIN-gpt-5-mini`
- embedding dimensions: `1536`
- vector database: Pinecone

Chunk size is implemented as an approximate token count using whitespace-separated words. This keeps chunks below the assignment limit while avoiding an extra tokenizer dependency.

## API

The root URL serves a small test UI. The graded API endpoints are:

### `POST /api/prompt`

Input:

```json
{
  "question": "Your natural language question here"
}
```

Output:

```json
{
  "response": "Final natural language answer from the model.",
  "context": [
    {
      "article_id": "1234",
      "title": "Sample article title",
      "chunk": "article chunk retrieved",
      "score": 0.1234
    }
  ],
  "Augmented_prompt": {
    "System": "the system prompt used to query the chat model",
    "User": "the user prompt used to query the chat model"
  }
}
```

### `GET /api/stats`

Output:

```json
{
  "chunk_size": 512,
  "overlap_ratio": 0.2,
  "top_k": 7
}
```

## Local Setup

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Set environment variables from `.env.example`. The course API file uses:

- `LLMOD_API_KEY`
- `LLMOD_BASE_URL`
- `LLMOD_CHAT_MODEL`
- `LLMOD_EMBEDDING_MODEL`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`

## Ingest Data

Start with a cheap subset:

```bash
python3 scripts/ingest.py --limit 100
```

Then test the four assignment-style questions:

```bash
python3 scripts/test_queries.py
```

After the subset works, ingest the full dataset once:

```bash
python3 scripts/ingest.py
```

Do not re-embed the full corpus for every code change. Change prompts and API code without touching Pinecone unless chunking parameters change.

## Deployment

Deploy to Vercel and set the same environment variables in the Vercel project settings. Submit:

- public live URL
- public GitHub URL

Keep the Pinecone index active until grading is complete.

## Environment Check

Before ingesting the full corpus, verify the keys:

```bash
python3 scripts/check_env.py
```

This script prints only whether keys are set and their lengths. It does not print secret values.
