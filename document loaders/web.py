from langchain_community.document_loaders import WebBaseLoader

url = "https://www.apple.com/in/macbook-neo/?afid=p240%7Cgo~cmp-11116556120~adg-109516736099~ad-816969290815_kwd-70212086~dev-c~ext-337057058263~prd-~mca-~nt-search&cid=aos-in-kwgo-txt-brand-brand--"

data = WebBaseLoader(url)
docs = data.load()

#print(len(docs))
print(docs[0].page_content)