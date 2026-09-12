from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

data = PyPDFLoader("document loaders/mern.pdf")
docs = data.load()

template = ChatPromptTemplate.from_messages(
    [("system", "you are a AI that summarizes the text"), 
     ("human", "{data}")]
)

model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash")
prompt = template.format_messages(data = docs[0].page_content)

result = model.invoke(prompt)

print(result.content[0]["text"])
