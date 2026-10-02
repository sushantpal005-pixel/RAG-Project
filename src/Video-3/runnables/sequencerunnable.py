from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint

# 1. prompt template
prompt = ChatPromptTemplate.from_template("Explain {topic} in simple words")

# 2. model

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
)
model = ChatHuggingFace(
    llm=llm
)

# 3. output parser
parser =  StrOutputParser()

# # step by step manual flow
# formatted_prompt = prompt.format_messages(topic="Machine Learning")

# # call the model manually
# response = model.invoke(formatted_prompt)

# # parse the output manually 
# final_output = parser.parse(response.content)

#print(final_output)

chain = prompt | model | parser

result = chain.invoke("Machine Learning")
print(result)