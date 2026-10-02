import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

# code_prompt and explain_prompt
code_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a code generator"),
    ("human", "{topic}")
])

explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistant who explain code in simple terms"),
    ("human","Explain the follwing code in  simple words:\n{code}")
])

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
)
model = ChatHuggingFace(
    llm=llm
)

parser = StrOutputParser()


# seq = code_prompt | model | parser | explain_prompt | model | parser
# result = seq.invoke({"topic": "write a code of palindrome in python"})
# print(result)

seq = code_prompt | model | parser
seq2 = RunnableParallel(
    {"code": RunnablePassthrough(),
      "explanation": explain_prompt | model | parser
    }
)

chain = seq | seq2

result = chain.invoke({"topic": "write a code of palindrome in java"})

print(result["code"])
print(result["explanation"])