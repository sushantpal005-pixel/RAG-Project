from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()


template = ChatPromptTemplate.from_messages(
    [("system", "you are a AI that summarizes the text"), 
     ("human", "{data}")]
)

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    # temperature=0.7,
    max_new_tokens=2048
    
)

model = ChatHuggingFace(llm=llm)
prompt = template.format_messages(data = docs[0].page_content)

result = model.invoke(prompt)

print(result.content)
