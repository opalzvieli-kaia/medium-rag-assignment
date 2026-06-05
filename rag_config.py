import os


CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "512"))
OVERLAP_RATIO = float(os.getenv("RAG_OVERLAP_RATIO", "0.2"))
TOP_K = int(os.getenv("RAG_TOP_K", "8"))

EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "medium-articles")

CHAT_MODEL = os.getenv("LLMOD_CHAT_MODEL", os.getenv("OPENAI_CHAT_MODEL", "4UHRUIN-gpt-5-mini"))
EMBEDDING_MODEL = os.getenv(
    "LLMOD_EMBEDDING_MODEL",
    os.getenv("OPENAI_EMBEDDING_MODEL", "4UHRUIN-text-embedding-3-small"),
)

SYSTEM_PROMPT = """You are a Medium-article assistant that answers questions strictly and only based on the Medium articles dataset context provided to you (metadata and article passages). You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided context, respond: "I don't know based on the provided Medium articles data."

Always explain your answer using the given context, quoting or paraphrasing the relevant article passage or metadata when helpful.

For requests asking for multiple article titles, return distinct articles, not repeated chunks from the same article. Follow exact formatting constraints in the user's question when they ask for them, such as "return only the titles" or "exactly 3 articles"."""


def validate_config():
    if CHUNK_SIZE > 1024:
        raise ValueError("RAG_CHUNK_SIZE must be at most 1024")
    if not 0 <= OVERLAP_RATIO <= 0.3:
        raise ValueError("RAG_OVERLAP_RATIO must be between 0 and 0.3")
    if not 1 <= TOP_K <= 30:
        raise ValueError("RAG_TOP_K must be between 1 and 30")


validate_config()
