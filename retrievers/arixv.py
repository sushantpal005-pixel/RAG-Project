
# from langchain_community.retrievers import ArxivRetriever

# #create the retriever
# retriever = ArxivRetriever(
#     load_max_docs= 2,        #number of papers to retrieve
#     load_all_available_meta= True,
#     top_k_results=2
# )

# #query arxic
# docs = retriever.invoke("large language models")

# #print results
# for i, doc in enumerate(docs):
#     print(f"\nResult {i+1}")
#     print("Title:", doc.metadata.get("Title"))
#     print("Authors:", doc.metadata.get("Authors"))
#     print("Summary:", doc.page_content[:500]) #print first 500 characters

#wikipedia retrievers (campus x)
import wikipedia
wikipedia.wikipedia.USER_AGENT = (
    "RAGProject/1.0 (sushantpal00009@gmail.com)"
)

from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=2, lang="en")

query = "famous israeli spy"

try:
    docs = retriever.invoke(query)
    for i, doc in enumerate(docs):
        print(f"\n--- Result {i+1} ---")
        print(f"Content:\n{doc.page_content}...")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

