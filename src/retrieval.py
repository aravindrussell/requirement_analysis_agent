import re
import uuid
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import EMBEDDING_MODEL

CHROMA_DIR = str(Path("data/chroma"))

def build_vector_store(documents):
    """Split loaded documents and persist their embeddings in Chroma."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No readable text was found in the uploaded documents.")

    # Ensure each chunk carries a readable source label.
    for chunk in chunks:
        source = chunk.metadata.get("source", "Unknown document")
        page = chunk.metadata.get("page")
        chunk.metadata["source_label"] = (
            f"{source}, page {page + 1}" if isinstance(page, int)
            else source
        )

    collection_name = "req_" + uuid.uuid4().hex[:16]
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_DIR,
    )
    return store, len(chunks)

def retrieve_context(vector_store, question: str, k: int = 5):
    docs = vector_store.similarity_search(question, k=k)
    if not docs:
        return "", []

    blocks = []
    sources = []
    for index, doc in enumerate(docs, start=1):
        label = doc.metadata.get("source_label", doc.metadata.get("source", "Unknown"))
        blocks.append(f"[SOURCE {index}: {label}]\n{doc.page_content}")
        sources.append({
            "label": label,
            "excerpt": doc.page_content[:500],
        })

    return "\n\n".join(blocks), sources
