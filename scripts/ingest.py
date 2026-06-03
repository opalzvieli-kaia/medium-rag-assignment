import argparse
import ast
import csv
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinecone import Pinecone, ServerlessSpec

from rag_config import (
    CHUNK_SIZE,
    EMBEDDING_DIMENSIONS,
    OVERLAP_RATIO,
    PINECONE_NAMESPACE,
)
from rag_core import batched, embed_texts


def parse_authors(raw: str) -> str:
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return ", ".join(str(author) for author in parsed if str(author).strip())
    except Exception:
        pass
    return raw


def chunk_words(text: str, chunk_size: int, overlap_ratio: float) -> List[str]:
    words = text.split()
    if not words:
        return []
    overlap = int(chunk_size * overlap_ratio)
    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


def article_id_for(row: Dict, row_number: int) -> str:
    stable = f"{row_number}|{row.get('title', '')}|{row.get('url', '')}"
    return hashlib.sha1(stable.encode("utf-8")).hexdigest()[:16]


def iter_chunks(csv_path: Path, limit: int = 0) -> Iterable[Dict]:
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row_number, row in enumerate(reader):
            if limit and row_number >= limit:
                break
            article_id = article_id_for(row, row_number)
            text = row["text"]
            title = row["title"]
            authors = parse_authors(row["authors"])
            tags = row["tags"]
            url = row["url"]
            timestamp = row["timestamp"]
            for chunk_id, chunk in enumerate(chunk_words(text, CHUNK_SIZE, OVERLAP_RATIO)):
                yield {
                    "id": f"{article_id}-{chunk_id}",
                    "text_for_embedding": (
                        f"Title: {title}\nAuthors: {authors}\nTags: {tags}\n\n{chunk}"
                    ),
                    "metadata": {
                        "article_id": article_id,
                        "chunk_id": chunk_id,
                        "title": title,
                        "authors": authors,
                        "url": url,
                        "timestamp": timestamp,
                        "tags": tags,
                        "chunk": chunk,
                    },
                }


def ensure_index(index_name: str, cloud: str, region: str):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        while not pc.describe_index(index_name).status["ready"]:
            print(f"Waiting for Pinecone index '{index_name}' to be ready...")
            time.sleep(5)
    return pc.Index(index_name)


def main():
    parser = argparse.ArgumentParser(description="Embed Medium CSV chunks into Pinecone.")
    parser.add_argument("--csv", default="medium-english-50mb.csv")
    parser.add_argument("--limit", type=int, default=0, help="Limit articles for cheap testing.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cloud", default=os.getenv("PINECONE_CLOUD", "aws"))
    parser.add_argument("--region", default=os.getenv("PINECONE_REGION", "us-east-1"))
    parser.add_argument(
        "--reset-namespace",
        action="store_true",
        help="Delete all vectors in the namespace before ingesting.",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only count chunks that would be ingested; do not call external APIs.",
    )
    args = parser.parse_args()

    if args.count_only:
        total_chunks = sum(1 for _ in iter_chunks(Path(args.csv), args.limit))
        print(
            f"Would ingest {total_chunks} chunks "
            f"(chunk_size={CHUNK_SIZE}, overlap_ratio={OVERLAP_RATIO}, namespace={PINECONE_NAMESPACE})"
        )
        return

    index_name = os.getenv("PINECONE_INDEX_NAME", os.getenv("PINECONE_INDEX"))
    if not index_name:
        raise RuntimeError("Set PINECONE_INDEX_NAME before ingesting.")

    index = ensure_index(index_name, args.cloud, args.region)
    if args.reset_namespace:
        print(f"Deleting existing vectors in namespace '{PINECONE_NAMESPACE}'...", flush=True)
        index.delete(delete_all=True, namespace=PINECONE_NAMESPACE)
        print("Namespace reset complete.", flush=True)

    total = 0
    batch = []
    print(
        f"Starting ingestion "
        f"(chunk_size={CHUNK_SIZE}, overlap_ratio={OVERLAP_RATIO}, namespace={PINECONE_NAMESPACE})",
        flush=True,
    )
    for item in iter_chunks(Path(args.csv), args.limit):
        batch.append(item)
        if len(batch) < args.batch_size:
            continue

        embeddings = embed_texts([item["text_for_embedding"] for item in batch])
        vectors = [
            {
                "id": item["id"],
                "values": embedding,
                "metadata": item["metadata"],
            }
            for item, embedding in zip(batch, embeddings)
        ]
        index.upsert(vectors=vectors, namespace=PINECONE_NAMESPACE)
        total += len(vectors)
        print(f"Upserted {total} chunks", flush=True)
        batch = []

    if batch:
        embeddings = embed_texts([item["text_for_embedding"] for item in batch])
        vectors = [
            {
                "id": item["id"],
                "values": embedding,
                "metadata": item["metadata"],
            }
            for item, embedding in zip(batch, embeddings)
        ]
        index.upsert(vectors=vectors, namespace=PINECONE_NAMESPACE)
        total += len(vectors)
        print(f"Upserted {total} chunks", flush=True)

    print(f"Done. Total chunks upserted: {total}", flush=True)


if __name__ == "__main__":
    main()
