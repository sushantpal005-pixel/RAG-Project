from dotenv import load_dotenv
load_dotenv()
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

search_tool = TavilySearchResults(max_result = 5)


llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
)
model = ChatHuggingFace(
    llm=llm
)

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assitant

summarize the following news into clear bullet points

{news}
"""
)

chain = prompt | model | StrOutputParser()

news_result = search_tool.run("Latest AI news of 2026")

result = chain.invoke({"news": news_result})

print(result)


print(search_tool.name)
print(search_tool.description)
print(search_tool.args)