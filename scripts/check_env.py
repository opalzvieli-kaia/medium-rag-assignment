import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinecone import Pinecone

from rag_core import embed_texts


def present(name: str) -> str:
    value = os.getenv(name, "")
    return f"{name}: set={bool(value)} len={len(value)}"


print(present("LLMOD_API_KEY"))
print(present("LLMOD_BASE_URL"))
print(present("LLMOD_CHAT_MODEL"))
print(present("LLMOD_EMBEDDING_MODEL"))
print(present("PINECONE_API_KEY"))
print(present("PINECONE_INDEX_NAME"))

if os.getenv("LLMOD_API_KEY"):
    vector = embed_texts(["Small environment check for the Medium RAG assignment."])[0]
    print(f"embedding_ok=True dimensions={len(vector)}")

if os.getenv("PINECONE_API_KEY"):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    indexes = pc.list_indexes().names()
    print(f"pinecone_ok=True indexes={indexes}")
