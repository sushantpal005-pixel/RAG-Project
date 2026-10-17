from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

#step - 1: document sources
documents = [
    Document(page_content="Langchain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="Hugging Face provide powerful embeddings models."),
    ]

#step - 2: initializing embedding model
embedding_model = HuggingFaceEmbeddings()

#step - 3: Create chroma vector store in memory
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="my_collection"
)

#step - 4: convert vectorstore into a retriever
retriver = vectorstore.as_retriever(search_kwags={"k": 2})

query = "what is Chroma used for ?"
results = retriver.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)