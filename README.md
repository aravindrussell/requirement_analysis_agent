# AI Requirement Analysis Agent

A Streamlit application that indexes requirement documents and answers questions
using either an OpenAI model or an Anthropic Claude model through LangChain.

## Features

- Upload PDF, DOCX, TXT and Markdown requirements.
- Extract document text and split it into overlapping chunks.
- Create embeddings and store them in a local Chroma vector database.
- Select OpenAI or Claude and enter a model ID.
- Retrieve relevant excerpts and answer with source labels.
- Preserve chat messages during the Streamlit session.

## Important API billing note

This starter uses OpenAI `text-embedding-3-small` embeddings for document indexing.
That means an OpenAI API key and API billing are needed even when Claude is selected
for answer generation. Claude calls are billed separately by Anthropic.

## Setup on Windows PowerShell

```powershell
cd C:\AI_Testing
# If you extracted the project elsewhere, change to that folder.
cd requirement_analysis_agent

python -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

Copy-Item .env.example .env
notepad .env
```

Add your real API keys to `.env`. Do not commit or share this file.

Start the application:

```powershell
python -m streamlit run app.py
```

Streamlit will print a local URL, usually http://localhost:8501.

## Model IDs

Defaults in `.env.example` are examples. Model access varies by API account.
If a model is unavailable, replace its ID with one listed in the relevant provider's
current API model catalog.

- OpenAI models: https://developers.openai.com/api/docs/models
- Claude models: https://platform.claude.com/docs/en/models/overview

## Architecture / concepts

1. **Document loader**: reads PDF, DOCX, TXT and Markdown files into LangChain Documents.
2. **Chunking**: splits long documents into overlapping passages so retrieval can find
   relevant sections without passing the whole document to the LLM.
3. **Embeddings**: converts text into vectors. This project uses OpenAI embeddings.
4. **Vector database**: Chroma stores vectors and text metadata locally.
5. **Retrieval**: similarity search finds the chunks most relevant to a question.
6. **Prompt**: gives the LLM the retrieved context and rules against unsupported claims.
7. **Model factory**: uses LangChain's `init_chat_model` to create the selected provider.
8. **Streamlit session state**: retains the vector store and chat messages across reruns.

## Prototype limitations

- The current implementation creates a new Chroma collection when you process a
  document set; old collections remain on disk until manually cleaned up.
- Requirement citations use source filename and page metadata where available.
  They are retrieval references, not proof that every answer is correct.
- It does not yet support authentication, multi-user isolation, automatic evaluation,
  requirement traceability matrices, or autonomous tool calling.
- Treat uploaded documents as untrusted data and do not upload confidential company
  requirements unless your organization's data policy permits the selected API providers.
