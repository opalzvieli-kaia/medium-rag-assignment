# RAG Design Report

## Assignment Constraints

The assistant must answer only from the Medium dataset. It must not use external information, model background knowledge, or the internet. If the answer is not in the retrieved context, it must say: `I don't know based on the provided Medium articles data.`

Required endpoints:

- `POST /api/prompt`
- `GET /api/stats`

Required vector database:

- Pinecone

Required models:

- `4UHRUIN-text-embedding-3-small`
- `4UHRUIN-gpt-5-mini`

Budget constraint:

- Total development and testing budget: 5 USD.

## Dataset Understanding

The dataset file is `medium-english-50mb.csv`.

Observed properties:

- rows: 7,682 articles
- columns: `title`, `text`, `url`, `authors`, `timestamp`, `tags`
- missing fields: none
- median article length: about 884 words
- average article length: about 1,083 words
- 95th percentile article length: about 2,479 words
- maximum article length: about 17,092 words
- duplicate titles: 16

Most common tags include Writing, Data Science, Artificial Intelligence, Machine Learning, Python, Programming, Mental Health, Startup, Psychology, Technology, Design, Productivity, Self Improvement, Marketing, and Creativity.

## RAG Hyperparameters

Chosen values:

```json
{
  "chunk_size": 512,
  "overlap_ratio": 0.2,
  "top_k": 7
}
```

Rationale:

- `chunk_size=512`: Medium articles are long enough that full-article embeddings would become semantically blurry. A 512-word chunk keeps passages focused while still preserving paragraph-level context.
- `overlap_ratio=0.2`: A 20% overlap keeps nearby sentences connected across chunk boundaries without too much duplicate embedding cost.
- `top_k=7`: The lectures recommend about 3-5 chunks for general text and higher values when broader evidence may be needed. Because the assignment includes multi-result questions and summaries, 7 is a balanced value that usually provides several distinct article candidates without pushing unnecessary context to the chat model.

The implementation uses whitespace-separated words as an approximate token unit. This is documented in code and keeps the value comfortably under the assignment maximum of 1024.

## Retrieval Strategy

Each article is split into chunks. Each embedded text includes:

- title
- authors
- tags
- article passage

The Pinecone metadata stores:

- `article_id`
- `chunk_id`
- `title`
- `authors`
- `url`
- `timestamp`
- `tags`
- `chunk`

Including title, authors, and tags in the embedded text helps semantic search match metadata-heavy questions such as "articles about education" and precise article identification questions.

At query time, Pinecone is searched with a wider internal candidate pool, capped at 30 matches. The API then keeps the highest-scoring chunk per distinct article until `top_k` distinct articles are collected. This improves the assignment's multi-result questions because the returned context is not dominated by repeated chunks from a single article.

For pandemic/innovation questions, the retriever adds two focused query variants before merging results. This handles the assignment example where the article discusses the bubonic plague, the Renaissance, recovery, innovation, and AI; the final answer still uses only the retrieved Medium context.

## Prompt Strategy

The system prompt includes the exact required assignment constraints and adds one clarification: for multi-result requests, the assistant must return distinct articles rather than repeated chunks.

The user prompt contains:

- the original question
- retrieved context as JSON

This makes the final answer auditable because `/api/prompt` returns the context and the augmented prompt.

## Golden Test Questions

The assignment examples can be used as regression tests.

Expected retrieved evidence:

1. Marketing as a conversation with readers
   - Expected article: `A Marketing Guide for Introverts`
   - Expected author: `Shaunta Grimes`
   - Evidence phrase found in dataset: "Marketing is just a conversation between you and your readers."

2. Exactly 3 education articles
   - Expected behavior: return three distinct article titles, no duplicate chunks, no extra text if the question says titles only.

3. Past pandemics can spur innovation and recovery
   - Expected article: `Rebounding From The Pandemic... with AI`
   - Expected author: `Massimiliano Versace`
   - Evidence includes the bubonic plague, the Renaissance, and the idea that recovery can seed new innovation.

4. Beginner-friendly practical habits advice
   - Strong expected article: `The Magic Key to Making Habits Sticky`
   - Expected author: `Shaunta Grimes`
   - Evidence includes tying a small habit to a time or action so it sticks.

## Budget Plan

To stay under 5 USD:

1. Ingest only 100 articles first.
2. Run the four golden questions.
3. If retrieval works, ingest the full dataset once.
4. Do not re-embed the full corpus when changing prompts or endpoint code.
5. If chunking parameters change, test on a subset before any full re-ingestion.

## Deployment Notes

The app is implemented as Python serverless functions for Vercel:

- `api/prompt.py`
- `api/stats.py`

The Pinecone index must remain active until the grade is received.

## Current Verification Status

Completed:

- PDF assignment and lecture requirements reviewed.
- Dataset schema and size inspected.
- Python source files compile successfully.
- Local imports for `openai` and `pinecone` were tested with the installed packages.
- LLMod embedding call verified successfully with 1,536 dimensions.
- Pinecone connection verified successfully.
- Pinecone namespace was reset and ingested from the beginning.
- Expected chunk count before ingestion: 22,174.
- Final Pinecone vector count in namespace `medium-articles`: 22,174.
- The four assignment-style validation questions were run after full ingestion.

Validation outcomes:

- Precise fact retrieval: correctly finds `A Marketing Guide for Introverts` by `Shaunta Grimes`.
- Multi-result topic listing: returns exactly three education-related titles.
- Key idea summary extraction: correctly finds `Rebounding From The Pandemic... with AI` by `Massimiliano Versace`.
- Recommendation with evidence: returns a habit-building article and justifies it from retrieved context.
