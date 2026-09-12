from langchain_community.document_loaders import TextLoader

data = TextLoader("document loaders/mern.txt")

docs = data.load()
print(len(docs))
