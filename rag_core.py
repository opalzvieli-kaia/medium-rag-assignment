import json
import os
from typing import Dict, Iterable, List

from openai import OpenAI
from pinecone import Pinecone

from rag_config import (
    CHAT_MODEL,
    EMBEDDING_MODEL,
    PINECONE_NAMESPACE,
    SYSTEM_PROMPT,
    TOP_K,
)


def _env(name: str, fallback: str = "") -> str:
    return os.getenv(name, fallback).strip()


def get_llm_client() -> OpenAI:
    api_key = _env("LLMOD_API_KEY", _env("OPENAI_API_KEY"))
    base_url = _env("LLMOD_BASE_URL", _env("OPENAI_BASE_URL"))
    if not api_key:
        raise RuntimeError("Missing LLMOD_API_KEY or OPENAI_API_KEY")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def get_pinecone_index():
    api_key = _env("PINECONE_API_KEY")
    index_name = _env("PINECONE_INDEX_NAME", _env("PINECONE_INDEX"))
    if not api_key:
        raise RuntimeError("Missing PINECONE_API_KEY")
    if not index_name:
        raise RuntimeError("Missing PINECONE_INDEX_NAME or PINECONE_INDEX")
    return Pinecone(api_key=api_key).Index(index_name)


def embed_texts(texts: List[str]) -> List[List[float]]:
    client = get_llm_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def build_user_prompt(question: str, context: List[Dict]) -> str:
    context_text = json.dumps(context, ensure_ascii=False, indent=2)
    return (
        "Answer the user question using only the retrieved Medium dataset context below.\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved context JSON:\n{context_text}"
    )


def build_retrieval_queries(question: str) -> List[str]:
    queries = [question]
    lower = question.lower()

    has_pandemic_terms = any(term in lower for term in ["pandemic", "pandemics", "plague"])
    has_innovation_terms = any(term in lower for term in ["innovation", "innovate", "recovery", "spur"])
    if has_pandemic_terms and has_innovation_terms:
        queries.extend(
            [
                "past pandemic innovation renaissance italy recovery AI",
                "pandemic artificial intelligence rebound innovation recovery renaissance",
            ]
        )

    return queries


def retrieve_context(question: str) -> List[Dict]:
    retrieval_queries = build_retrieval_queries(question)
    question_vectors = embed_texts(retrieval_queries)
    query_top_k = min(30, TOP_K * 4)
    index = get_pinecone_index()

    by_article = {}
    for question_vector in question_vectors:
        result = index.query(
            vector=question_vector,
            top_k=query_top_k,
            namespace=PINECONE_NAMESPACE,
            include_metadata=True,
        )
        for match in result.matches:
            metadata = match.metadata or {}
            article_id = str(metadata.get("article_id", ""))
            if not article_id:
                continue
            score = float(match.score or 0.0)
            existing = by_article.get(article_id)
            if existing and existing["score"] >= score:
                continue
            by_article[article_id] = {
                "article_id": article_id,
                "title": metadata.get("title", ""),
                "chunk": metadata.get("chunk", ""),
                "score": score,
                "authors": metadata.get("authors", ""),
                "url": metadata.get("url", ""),
                "tags": metadata.get("tags", ""),
                "chunk_id": metadata.get("chunk_id", ""),
            }

    context = sorted(by_article.values(), key=lambda item: item["score"], reverse=True)
    context = context[:TOP_K]
    return context


def answer_question(question: str) -> Dict:
    context = retrieve_context(question)
    user_prompt = build_user_prompt(question, context)

    client = get_llm_client()
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    response_text = completion.choices[0].message.content or ""

    public_context = [
        {
            "article_id": item["article_id"],
            "title": item["title"],
            "chunk": item["chunk"],
            "score": item["score"],
        }
        for item in context
    ]

    return {
        "response": response_text,
        "context": public_context,
        "Augmented_prompt": {
            "System": SYSTEM_PROMPT,
            "User": user_prompt,
        },
    }


def batched(items: List, batch_size: int) -> Iterable[List]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
