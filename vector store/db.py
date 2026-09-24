from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv() 

from langchain_core.documents import Document

docs = [
    Document(page_content="python is widely used in Artificial Inntelligence", metadata={"source": "AI_book"}),
    Document(page_content="pandas is used for data analytics in python", metadata={"source": "DataScience_book"}),
    Document(page_content="Nerual networks are used in deep learning", metadata={"source": "DL_book"})
]

embedding_model = HuggingFaceEmbeddings(
    model_name= "sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents= docs,
    embedding= embedding_model,
    persist_directory= "chroma-db"
)

result = vectorstore.similarity_search("what is used for data analytics?", k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)

retriever = vectorstore.as_retriever()

docs = retriever.invoke("Explain deep learning")

for d in docs:
    print(d.page_content)