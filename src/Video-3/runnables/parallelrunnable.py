from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

# components
llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
)

model = ChatHuggingFace(
    llm=llm
)
parser = StrOutputParser()

# Two different prompts
short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 1-2 lines"
)

detailed_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in detail"
)

#Input
topic = "Machine Learning"

chain = RunnableParallel({
    "short": RunnableLambda(lambda x : x ['short']) | short_prompt | model | parser,
    "detailed" : RunnableLambda(lambda x : x['detailed']) | detailed_prompt | model | parser
})

result = chain.invoke({
    "short": {"topic": "Machine Learning"},
    "detailed": {"topic": "Deep Learning"}
})

print(result['short'])
print(result['detailed'])
# formatted_short = short_prompt.format_messages(topic = topic)
# response_short = model.invoke(formatted_short)
# str_out = parser.parse(response_short.content)

