
import os
import shutil
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

load_dotenv()

CHROMA_DIR = "chroma_db"
UPLOAD_DIR = "uploaded_books"

os.makedirs(UPLOAD_DIR, exist_ok=True)


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="BookMind - RAG",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #888888;
        margin-bottom: 30px;
    }

    .book-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #333333;
        margin-bottom: 20px;
    }

    .answer-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #444444;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
    '<div class="main-title">📚 BookMind</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Chat with your books using RAG + AI</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource
def get_embedding_model():

    return HuggingFaceEmbeddings()


embedding_model = get_embedding_model()


# --------------------------------------------------
# LLM
# --------------------------------------------------

@st.cache_resource
def get_llm():

    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V4-Flash-0731"
    )

    return ChatHuggingFace(llm=llm)


model = get_llm()


# --------------------------------------------------
# PROMPT
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful AI assistant.

            Use ONLY the provided context to answer the question.

            If the answer is not present in the context,
            say exactly:

            "I could not find the answer in the document."

            Do not use outside knowledge.
            """
        ),
        (
            "human",
            """
            Context:
            {context}

            Question:
            {question}
            """
        )
    ]
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("📖 Book")

    uploaded_file = st.file_uploader(
        "Upload your book",
        type=["pdf"]
    )

    st.divider()

    st.caption(
        "Upload a PDF book and ask questions about its content."
    )


# --------------------------------------------------
# CREATE VECTOR DATABASE
# --------------------------------------------------

if uploaded_file is not None:

    # Save uploaded PDF
    pdf_path = os.path.join(
        UPLOAD_DIR,
        uploaded_file.name
    )

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


    # Store currently uploaded book
    current_book = st.session_state.get("current_book")


    # Only process the book if it is a new upload
    if current_book != uploaded_file.name:

        with st.spinner("Processing your book..."):

            try:

                # -----------------------------
                # Load PDF
                # -----------------------------

                loader = PyPDFLoader(pdf_path)

                docs = loader.load()


                # -----------------------------
                # Split into chunks
                # -----------------------------

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = splitter.split_documents(docs)


                # -----------------------------
                # Remove old database
                # -----------------------------

                if os.path.exists(CHROMA_DIR):
                    shutil.rmtree(CHROMA_DIR)


                # -----------------------------
                # Create Chroma DB
                # -----------------------------

                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding_model,
                    persist_directory=CHROMA_DIR
                )


                # Save current book name
                st.session_state.current_book = uploaded_file.name

                # Save vectorstore
                st.session_state.vectorstore = vectorstore

                st.session_state.messages = []


                st.success(
                    f"Successfully processed {uploaded_file.name}"
                )

            except Exception as e:

                st.error(
                    f"Error processing PDF: {str(e)}"
                )


# --------------------------------------------------
# LOAD EXISTING VECTORSTORE
# --------------------------------------------------

if "vectorstore" not in st.session_state:

    if os.path.exists(CHROMA_DIR):

        st.session_state.vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embedding_model
        )


# --------------------------------------------------
# CHAT
# --------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

query = st.chat_input(
    "Ask something about your book..."
)


if query:

    # User message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)


    # Check database
    if "vectorstore" not in st.session_state:

        with st.chat_message("assistant"):

            st.warning(
                "Please upload a PDF book first."
            )

    else:

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                # -----------------------------
                # Retriever
                # -----------------------------

                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 4,
                        "fetch_k": 10,
                        "lambda_mult": 0.5
                    }
                )


                # -----------------------------
                # Retrieve documents
                # -----------------------------

                docs = retriever.invoke(query)


                # -----------------------------
                # Create context
                # -----------------------------

                context = "\n\n".join(
                    [
                        doc.page_content
                        for doc in docs
                    ]
                )


                # -----------------------------
                # Create prompt
                # -----------------------------

                final_prompt = prompt.invoke(
                    {
                        "context": context,
                        "question": query
                    }
                )


                # -----------------------------
                # Generate answer
                # -----------------------------

                response = model.invoke(
                    final_prompt
                )


                answer = response.content


                # -----------------------------
                # Display answer
                # -----------------------------

                st.markdown(answer)


                # -----------------------------
                # Save assistant message
                # -----------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

