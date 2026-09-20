from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

#token splitter
# splitter = TokenTextSplitter(
#     chunk_size = 1000,
#     chunk_overlap = 10
# )

#recursive text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 10
)

data = PyPDFLoader("document loaders/mern.pdf")

docs = data.load()

chunks = splitter.split_documents(docs)
print(chunks[0].page_content)

#print(docs[49])