from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("document loaders/mern.pdf")

docs = data.load()

print(docs[49])