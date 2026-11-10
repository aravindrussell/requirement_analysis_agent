from pathlib import Path
from tempfile import NamedTemporaryFile

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

def load_uploaded_file(uploaded_file):
    """Save uploaded bytes temporarily and load them as LangChain Documents."""
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {suffix}. Upload PDF, DOCX, TXT or MD."
        )

    temp_path = None
    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(uploaded_file.getvalue())
            temp_path = temp.name

        if suffix == ".pdf":
            docs = PyPDFLoader(temp_path).load()
        elif suffix == ".docx":
            docs = Docx2txtLoader(temp_path).load()
        else:
            docs = TextLoader(temp_path, encoding="utf-8").load()

        for doc in docs:
            doc.metadata["source"] = uploaded_file.name

        return docs
    finally:
        if temp_path and Path(temp_path).exists():
            Path(temp_path).unlink()
