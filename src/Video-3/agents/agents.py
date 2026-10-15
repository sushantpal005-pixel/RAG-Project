#first step - loading all the libraries
from dotenv import load_dotenv
load_dotenv()

import os
import requests

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient
from rich import print

#Now lets create some tools

#Weather tool
@tool
def get_weather(city: str) -> str:
    """"Get Current weather of a city"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    response = requests.get(url)
    data = response.json()


    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Weather in {city}: {desc}, {temp}°C"

#Tavily news tool

TAVILY_API = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(TAVILY_API)


@tool
def get_news(city : str) -> str:
    """Get Latest news about the city"""

    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )

    results = response.get("results", [])

    if not results:
        return f"No news found for {city}"

    news_list = []

    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "No content")

        news_list.append(
            f"- {title}\n {url}\n {snippet[:100]}..."
        )

    return f"Latest news in {city}: \n\n" + "\n\n".join(news_list)


# creating llm

llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.getenv("GROK_API_KEY"))

tools = {"get_weather": get_weather, "get_news": get_news}

# binding llm with tool
llm_with_tool = llm.bind_tools([get_weather, get_news])


# AGENT LOOP - VERY IMPORTANT

Message = []

print("City Intelligence System")
print("Type 0 to Quit")

while True:
    user_input = str(input("You : "))
    if user_input.lower() == "exit":
        break

    query = HumanMessage(content=user_input)

    Message.append(query)

    while True:
        AI_Message = llm_with_tool.invoke(Message)

        Message.append(AI_Message)

        # if tool is required
        if AI_Message.tool_calls:
            for tool_call in AI_Message.tool_calls:
                tool_name = tool_call["name"]

                # HUMAN IN THE LOOP
                confirm = input(f"Agent wants to call {tool_name} Approve (yes/no)")

                if confirm.lower() != "yes":
                    print("tool call denied and I cannot get the latest information")
                    break

                # Execute tool
                tool_result = tools[tool_name].invoke(tool_call)

                Message.append(
                    ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
                )

            continue
        else:
            print("\n Final Answer: \n")
            print(AI_Message.content)    
            break


