
import os
import requests
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from tavily import TavilyClient

load_dotenv()

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="City AI Agent",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, #152340 0%, transparent 35%),
            radial-gradient(ellipse at 90% 10%, #182039 0%, transparent 30%),
            #080B12;
        color: #E8EDF8;
    }

    [data-testid="stSidebar"] {
        background: #0C111C;
        border-right: 1px solid #202A3B;
    }

    .hero {
        padding: 30px 32px;
        border: 1px solid #29354A;
        border-radius: 24px;
        background: linear-gradient(135deg, #151F33 0%, #101521 55%, #17152A 100%);
        margin-bottom: 24px;
    }

    .hero-label {
        color: #8DA9FF;
        font-size: 12px;
        letter-spacing: 2px;
        font-weight: 700;
        text-transform: uppercase;
    }

    .hero h1 {
        color: #F4F7FF;
        font-size: clamp(30px, 4vw, 44px);
        font-weight: 800;
        letter-spacing: -1.5px;
        margin: 12px 0 8px;
    }

    .hero p {
        color: #A5B1C7;
        font-size: 15px;
        line-height: 1.7;
        margin-bottom: 0;
    }

    .section-title {
        color: #E7EDFA;
        font-size: 17px;
        font-weight: 700;
        margin: 20px 0 12px;
    }

    .feature-card {
        padding: 18px;
        min-height: 132px;
        border: 1px solid #273249;
        border-radius: 18px;
        background: rgba(17, 24, 39, 0.85);
        margin-bottom: 8px;
    }

    .feature-icon {
        font-size: 25px;
        margin-bottom: 10px;
    }

    .feature-title {
        color: #F1F5FF;
        font-weight: 700;
        font-size: 14px;
        margin-bottom: 6px;
    }

    .feature-desc {
        color: #94A3B8;
        font-size: 12px;
        line-height: 1.6;
    }

    .status-pill {
        display: inline-block;
        border: 1px solid #275D4A;
        background: #102A23;
        color: #75E0B5;
        border-radius: 30px;
        padding: 6px 11px;
        font-size: 11px;
        font-weight: 600;
    }

    .stChatMessage {
        background: rgba(16, 22, 35, 0.85);
        border: 1px solid #202B3E;
        border-radius: 18px;
        padding: 14px;
        margin-bottom: 12px;
    }

    .stChatInput textarea {
        background: #101624 !important;
        color: #F1F5FF !important;
        border: 1px solid #344158 !important;
        border-radius: 16px !important;
    }

    .stButton > button {
        border-radius: 12px;
        border: 1px solid #303C52;
        background: #151D2C;
        color: #E6ECF8;
        font-weight: 600;
        min-height: 42px;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #819BFF;
        color: #FFFFFF;
        background: #202C42;
    }

    div[data-testid="stAlert"] {
        border-radius: 14px;
    }

    hr {
        border-color: #222C3D;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# WEATHER TOOL — ORIGINAL FUNCTIONALITY
# --------------------------------------------------

@tool
def get_weather(city: str) -> str:
    """Get current weather of a city."""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return "Error: OPENWEATHER_API_KEY is missing from .env."

    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={api_key}&units=metric"
    )

    try:
        response = requests.get(url, timeout=15)
        data = response.json()

        if str(data.get("cod")) != "200":
            return f"Error: {data.get('message', 'Could not fetch weather')}"

        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]

        return f"Weather in {city}: {desc}, {temp}°C"

    except requests.RequestException as exc:
        return f"Weather request failed: {exc}"


# --------------------------------------------------
# TAVILY NEWS TOOL — ORIGINAL FUNCTIONALITY
# --------------------------------------------------

@tool
def get_news(city: str) -> str:
    """Get latest news about the city."""

    tavily_key = os.getenv("TAVILY_API_KEY")

    if not tavily_key:
        return "Error: TAVILY_API_KEY is missing from .env."

    try:
        tavily_client = TavilyClient(api_key=tavily_key)

        response = tavily_client.search(
            query=f"latest news in {city}",
            search_depth="basic",
            max_results=3,
        )

        results = response.get("results", [])

        if not results:
            return f"No news found for {city}"

        news_list = []

        for result in results:
            title = result.get("title", "No title")
            url = result.get("url", "")
            snippet = result.get("content", "No content")

            news_list.append(
                f"- {title}\n{url}\n{snippet[:100]}..."
            )

        return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)

    except Exception as exc:
        return f"News request failed: {exc}"


# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROK_API_KEY"),
)


# --------------------------------------------------
# HUMAN APPROVAL MIDDLEWARE
# Approval is handled in the Streamlit browser.
# --------------------------------------------------

@wrap_tool_call
def human_approval(request, handler):
    """Pause execution until the user approves or denies a tool call."""

    tool_name = request.tool_call["name"]

    decision = interrupt(
        {
            "type": "tool_approval",
            "tool_name": tool_name,
            "arguments": request.tool_call.get("args", {}),
            "tool_call_id": request.tool_call["id"],
        }
    )

    if decision != "yes":
        return ToolMessage(
            content="Tool call denied by user.",
            tool_call_id=request.tool_call["id"],
        )

    return handler(request)


# --------------------------------------------------
# CREATE AGENT
# --------------------------------------------------

@st.cache_resource
def build_agent():
    return create_agent(
        llm,
        tools=[get_weather, get_news],
        system_prompt=(
            "You are a helpful city assistant. "
            "Help users with city weather and latest city news. "
            "Use the available tools when needed."
        ),
        middleware=[human_approval],
        checkpointer=InMemorySaver(),
    )


agent = build_agent()

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = None

if "thread_id" not in st.session_state:
    import uuid
    st.session_state.thread_id = str(uuid.uuid4())

if "last_result" not in st.session_state:
    st.session_state.last_result = None

agent_config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}


def process_result(result):
    """Update chat history and detect pending tool approval."""

    st.session_state.last_result = result

    interrupts = result.get("__interrupt__", ())

    if interrupts:
        interrupt_value = interrupts[0].value
        st.session_state.pending_approval = interrupt_value
        return

    st.session_state.pending_approval = None

    # Read the latest assistant response, not a tool result.
    messages = result.get("messages", [])

    for message in reversed(messages):
        if getattr(message, "type", "") == "ai":
            if message.content:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                    }
                )
            break


def resume_agent(decision):
    """Resume the interrupted agent with the user's decision."""

    with st.spinner("Agent is processing your decision..."):
        result = agent.invoke(
            Command(resume=decision),
            config=agent_config,
        )

    process_result(result)
    st.rerun()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 12px 0 22px;">
            <div style="font-size: 30px;">🌍</div>
            <div style="font-size: 21px; font-weight: 800; color: #F1F5FF;">
                City<span style="color:#8DA9FF;">Agent</span>
            </div>
            <div style="color:#8795AD; font-size:12px; margin-top:5px;">
                AI-powered city intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Quick actions")

    if st.button("🌤️  Check weather", use_container_width=True):
        st.session_state.suggested_prompt = (
            "What is the current weather in New Delhi?"
        )
        st.rerun()

    if st.button("📰  Latest city news", use_container_width=True):
        st.session_state.suggested_prompt = (
            "Get the latest news about New Delhi."
        )
        st.rerun()

    st.divider()

    st.markdown("#### Agent capabilities")
    st.markdown("🌦️  Real-time city weather")
    st.markdown("🗞️  Latest news via Tavily")
    st.markdown("🧠  Tool-calling LLM")
    st.markdown("🛡️  Human approval for tool calls")

    st.divider()

    st.markdown("#### Configuration")

    for label, env_name in [
        ("Groq LLM", "GROK_API_KEY"),
        ("OpenWeather", "OPENWEATHER_API_KEY"),
        ("Tavily Search", "TAVILY_API_KEY"),
    ]:
        configured = bool(os.getenv(env_name))
        symbol = "🟢" if configured else "🔴"
        st.markdown(f"{symbol} **{label}**")

    st.divider()

    if st.button("🗑️  Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_approval = None
        st.session_state.last_result = None

        import uuid
        st.session_state.thread_id = str(uuid.uuid4())

        st.rerun()

    st.markdown(
        """
        <div style="color:#65738A; font-size:11px; padding-top:20px;">
            Built with Streamlit · LangChain · Groq · Tavily
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# MAIN HERO
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-label">✦ Intelligent city assistant</div>
        <h1>Your city, understood.</h1>
        <p>
            Ask about the weather, explore the latest local headlines,
            or chat with your AI agent. Every tool call requires your approval.
        </p>
        <br>
        <span class="status-pill">● Agent ready</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# CAPABILITY CARDS
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">🌤️</div>
            <div class="feature-title">Live Weather</div>
            <div class="feature-desc">
                Current temperature and weather conditions for your city.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">📰</div>
            <div class="feature-title">City News</div>
            <div class="feature-desc">
                Discover recent headlines and source links with Tavily.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">🛡️</div>
            <div class="feature-title">Human in the Loop</div>
            <div class="feature-desc">
                Review and approve every tool call before execution.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-title">💬 Chat with your agent</div>',
            unsafe_allow_html=True)

# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if not st.session_state.messages and not st.session_state.pending_approval:
    st.markdown(
        """
        <div style="padding: 12px 0 20px; color:#95A3BB;">
            <b style="color:#E8EDF8;">Hello! 👋</b><br>
            I'm your City AI Agent. Ask me about a city's weather
            or its latest news to get started.
        </div>
        """,
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🌍"

    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


# --------------------------------------------------
# APPROVAL PANEL
# --------------------------------------------------

if st.session_state.pending_approval:
    approval = st.session_state.pending_approval
    tool_name = approval.get("tool_name", "unknown")
    arguments = approval.get("arguments", {})

    st.warning("🛡️ Human approval required")

    st.markdown(f"**The agent wants to call:** `{tool_name}`")
    st.json(arguments)

    st.caption(
        "The tool will not execute until you approve it. "
        "Denying the request returns a denial message to the agent."
    )

    approve_col, deny_col = st.columns(2)

    with approve_col:
        if st.button(
            "✅ Approve tool call",
            type="primary",
            use_container_width=True,
        ):
            resume_agent("yes")

    with deny_col:
        if st.button("❌ Deny tool call", use_container_width=True):
            resume_agent("no")


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

suggested_prompt = st.session_state.pop("suggested_prompt", None)

if suggested_prompt and not st.session_state.pending_approval:
    user_prompt = suggested_prompt
else:
    user_prompt = None

if not st.session_state.pending_approval:
    typed_prompt = st.chat_input(
        "Ask about weather or news in any city...",
    )

    if typed_prompt:
        user_prompt = typed_prompt

if user_prompt and not st.session_state.pending_approval:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_prompt)

    with st.chat_message("assistant", avatar="🌍"):
        try:
            with st.spinner("Your city agent is thinking..."):
                result = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": user_prompt,
                            }
                        ]
                    },
                    config=agent_config,
                )

            process_result(result)
            st.rerun()

        except Exception as exc:
            st.error(f"Agent error: {exc}")

st.markdown("---")
st.caption(
    "City AI Agent · Powered by LangChain, Groq, OpenWeather and Tavily"
)
