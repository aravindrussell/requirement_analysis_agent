
import streamlit as st
from dotenv import load_dotenv

from src.config import (
    OPENAI_MODEL_DEFAULT,
    CLAUDE_MODEL_DEFAULT,
    has_openai_key,
    has_anthropic_key,
)
from src.document_loader import load_uploaded_file
from src.retrieval import build_vector_store
from src.model_factory import build_chat_model
from src.qa_agent import answer_question
from src.test_design import generate_test_design


load_dotenv()

st.set_page_config(
    page_title="AI Requirement Analysis Agent",
    page_icon="🧩",
    layout="wide",
)

st.title("🧩 AI Requirement Analysis Agent")
st.caption(
    "Upload software requirements, ask questions, and generate "
    "QA test scenarios and acceptance criteria using AI."
)


# --------------------------------------------------
# Sidebar: Model configuration
# --------------------------------------------------

with st.sidebar:
    st.header("Model configuration")

    provider = st.selectbox(
        "LLM provider",
        ["OpenAI", "Claude"],
    )

    if provider == "OpenAI":
        model_name = st.text_input(
            "OpenAI model ID",
            value=OPENAI_MODEL_DEFAULT,
            help="Use a model ID enabled for your OpenAI API account.",
        )
        key_available = has_openai_key()
        key_name = "OPENAI_API_KEY"

    else:
        model_name = st.text_input(
            "Claude model ID",
            value=CLAUDE_MODEL_DEFAULT,
            help="Use a model ID enabled for your Anthropic API account.",
        )
        key_available = has_anthropic_key()
        key_name = "ANTHROPIC_API_KEY"

    st.divider()

    st.markdown("**Retrieval configuration**")
    st.write("Embedding model: `text-embedding-3-small`")
    st.write("Vector database: `Chroma (local)`")

    st.caption(
        "OpenAI embeddings are used for retrieval even when Claude "
        "is selected. OpenAI API access and billing are required "
        "for document indexing."
    )


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "messages" not in st.session_state:
    st.session_state.messages = []

if "test_design_result" not in st.session_state:
    st.session_state.test_design_result = None

if "test_design_sources" not in st.session_state:
    st.session_state.test_design_sources = []


# --------------------------------------------------
# Section 1: Document upload and indexing
# --------------------------------------------------

st.header("1. Upload requirement documents")

uploads = st.file_uploader(
    "Choose PDF, DOCX, TXT or Markdown files",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
)

col_a, col_b = st.columns([1, 3])

with col_a:
    index_clicked = st.button(
        "Process documents",
        type="primary",
        disabled=not uploads,
        use_container_width=True,
    )

with col_b:
    st.caption(
        "Documents are split into chunks, embedded and stored in Chroma."
    )


if index_clicked:

    if not has_openai_key():
        st.error(
            "OPENAI_API_KEY is required for document embeddings. "
            "Add it to your .env file and restart Streamlit."
        )

    else:
        try:
            with st.spinner(
                "Reading documents, creating chunks and indexing..."
            ):
                all_docs = []

                for uploaded in uploads:
                    all_docs.extend(
                        load_uploaded_file(uploaded)
                    )

                store, count = build_vector_store(all_docs)

                st.session_state.vector_store = store
                st.session_state.uploaded_names = [
                    f.name for f in uploads
                ]
                st.session_state.chunk_count = count

                st.session_state.messages = []
                st.session_state.test_design_result = None
                st.session_state.test_design_sources = []

            st.success(
                f"Indexed {len(uploads)} document(s) "
                f"into {count} text chunks."
            )

        except Exception as exc:
            st.error("Document indexing failed.")
            st.exception(exc)


if st.session_state.vector_store is not None:

    st.success(
        f"Knowledge base ready: "
        f"{', '.join(st.session_state.uploaded_names)} "
        f"({st.session_state.chunk_count} chunks)"
    )

else:
    st.info(
        "Upload at least one requirement document and select "
        "Process documents."
    )


# --------------------------------------------------
# Section 2: Requirement question answering
# --------------------------------------------------

st.header("2. Ask a requirement question")

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("Retrieved requirement excerpts"):

                for source in message["sources"]:
                    st.markdown(f"**{source['label']}**")
                    st.write(source["excerpt"])


question = st.chat_input(
    "Example: What happens after five consecutive failed login attempts?",
    disabled=st.session_state.vector_store is None,
)


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        if not key_available:

            st.error(
                f"{key_name} is missing. Add it to your .env file "
                "and restart the app."
            )

        else:

            try:
                with st.spinner(f"Querying {provider}..."):

                    model = build_chat_model(
                        provider.lower(),
                        model_name.strip(),
                    )

                    answer, sources = answer_question(
                        model,
                        st.session_state.vector_store,
                        question,
                    )

                st.markdown(answer)

                if sources:

                    with st.expander("Retrieved requirement excerpts"):

                        for source in sources:
                            st.markdown(f"**{source['label']}**")
                            st.write(source["excerpt"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as exc:

                st.error(
                    "The model request failed. Check the selected "
                    "model ID, API key, billing and network connection."
                )
                st.exception(exc)


if st.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()


# --------------------------------------------------
# Section 3: AI Test Scenario and Test Case Generation
# --------------------------------------------------

st.divider()

st.header("3. Generate Test Scenarios & Test Cases")

st.write(
    "Generate functional test scenarios, detailed test cases, "
    "acceptance criteria, requirement gaps and QA recommendations "
    "from the indexed requirement documents."
)

st.caption(
    "The generator uses retrieved requirement content. "
    "Review all generated test cases before approving them."
)


if st.session_state.vector_store is None:

    st.info(
        "Upload and process requirement documents before generating "
        "test scenarios."
    )

else:

    generate_clicked = st.button(
        "Generate Test Design",
        type="primary",
        use_container_width=True,
    )

    if generate_clicked:

        if not key_available:

            st.error(
                f"{key_name} is missing. Configure your API key "
                "and restart the application."
            )

        else:

            try:
                with st.spinner(
                    f"Generating test design using {provider}..."
                ):

                    model = build_chat_model(
                        provider.lower(),
                        model_name.strip(),
                    )

                    design, sources = generate_test_design(
                        model,
                        st.session_state.vector_store,
                    )

                    st.session_state.test_design_result = design
                    st.session_state.test_design_sources = sources

            except Exception as exc:

                st.error(
                    "Test design generation failed. Check the model ID, "
                    "API key, provider billing and retrieved context."
                )
                st.exception(exc)


# Display generated output outside the button condition.
# This ensures results remain visible after Streamlit reruns.

if st.session_state.test_design_result:

    st.success("Test design generation completed.")

    st.markdown(
        st.session_state.test_design_result
    )

    if st.session_state.test_design_sources:

        with st.expander("Requirement sources used for test design"):

            for source in st.session_state.test_design_sources:
                st.markdown(f"**{source['label']}**")
                st.write(source["excerpt"])

    st.download_button(
        label="Download test design as Markdown",
        data=st.session_state.test_design_result,
        file_name="generated_test_design.md",
        mime="text/markdown",
        use_container_width=True,
    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Prototype: RAG-based requirement analysis and AI test design. "
    "Generated outputs require QA review and are not automatically "
    "approved or executed test cases."
)
