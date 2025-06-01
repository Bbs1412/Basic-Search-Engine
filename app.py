import streamlit as st

import os
from dotenv import load_dotenv
from typing import List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.agents import AgentType, initialize_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from langchain_core.language_models import BaseChatModel

# ----------------------------------------------------------------------------------------
# Page Setup and Initialization:
# ----------------------------------------------------------------------------------------

st.set_page_config(page_title="Agentic AI Search Engine", page_icon="🔍", layout="wide")
st.header("🔍 :orange[Basic Search Engine using Agentic AI]", divider=True)
load_dotenv()
# st.session_state.setdefault("temperature", 0.75)

inits = {
    "providers": ["Ollama", "OpenAI", "Google", "Groq"],
    # "temperature": 0.75,
    "provider": "Ollama",
    "model": "gemma3:latest",
    "user_api_key": None,
    "chat_history": [AIMessage("`Hello 👋!` I am Smart ChatBot who can search on web. How can I assist you today?")],
}

for key, value in inits.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ----------------------------------------------------------------------------------------
# Helpers:
# ----------------------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_model_list(provider: str) -> List[str]:
    """Returns the list of available models from given provider.

    Args:
        provider (str): The name of the AI provider (e.g., "OpenAI", "Groq", "Google", "Ollama").

    Returns:
        List[str]: A list of model IDs available for the specified provider.
    """
    # sleep(3)

    # Update API key in session state:
    st.session_state.user_api_key = get_api_key(provider)

    if provider == "Groq":
        from groq import Groq           # type: ignore
        client = Groq(api_key=st.secrets.Groq.API_KEY)
        return [str(model.id) for model in client.models.list().data]

    elif provider == "OpenAI":
        from openai import OpenAI       # type: ignore
        client = OpenAI(api_key=st.secrets.OpenAI.API_KEY)
        return [str(model.id) for model in client.models.list().data]

    elif provider == "Ollama":
        import ollama                   # type: ignore
        return [str(model.model) for model in ollama.list().models]

    elif provider == "Google":
        from google import genai        # type: ignore
        client = genai.Client(api_key=st.secrets.Google.API_KEY)
        return [str(model.name) for model in client.models.list()]

    else:
        return ["demo-model"]


def get_api_key(provider: str) -> str | None:
    """Returns the API key from Streamlit secrets or environment variables, if available.

    Args:
        provider (str): The name of the AI provider (e.g., "OpenAI", "Groq", "Google", "Ollama").

    Returns:
        str | None: The API key if found, otherwise None.
    """
    provider_map = {
        "OpenAI": ("OpenAI", "OPENAI_API_KEY"),
        "Groq": ("Groq", "GROQ_API_KEY"),
        "Google": ("Google", "GEMINI_API_KEY"),
    }

    secret_path, env_var = provider_map.get(provider, (None, None))
    if not secret_path or not env_var:
        return None

    # Try Streamlit secrets
    try:
        return st.secrets[secret_path]["API_KEY"]
    except Exception:
        pass

    # Try environment variable
    return os.getenv(env_var, None)


def get_llm_instance(provider: str, model: str, temperature: float = 0.75) -> BaseChatModel:
    """Returns an instance of the language model based on the provider and model name.

    Args:
        provider (str): The name of the AI provider (e.g., "OpenAI", "Groq", "Google", "Ollama").
        model (str): The model name to be used.
        temperature (float): The creativity level for the model's responses.

    Returns:
        BaseChatModel: An instance of the language model.
    """

    if st.session_state.provider == "Ollama":
        from langchain_ollama import ChatOllama                 # type: ignore
        st.session_state.llm = ChatOllama(
            model=st.session_state.model, temperature=st.session_state.temperature
        )

    elif st.session_state.provider == "OpenAI":
        from langchain_openai import ChatOpenAI                 # type: ignore
        st.session_state.llm = ChatOpenAI(
            model=st.session_state.model, temperature=st.session_state.temperature,
            api_key=st.session_state.user_api_key,
        )

    elif st.session_state.provider == "Groq":
        from langchain_groq import ChatGroq                     # type: ignore
        st.session_state.llm = ChatGroq(
            model=st.session_state.model, temperature=st.session_state.temperature,
            api_key=st.session_state.user_api_key,
        )

    elif st.session_state.provider == "Google":
        from langchain_google_genai import ChatGoogleGenerativeAI     # type: ignore
        st.session_state.llm = ChatGoogleGenerativeAI(
            model=st.session_state.model, temperature=st.session_state.temperature,
            api_key=st.session_state.user_api_key,
        )

    else:
        st.error("Unsupported provider selected. Please choose a valid provider and model.")
        st.stop()

    return st.session_state.llm


# ----------------------------------------------------------------------------------------
# Sidebar:
# ----------------------------------------------------------------------------------------

st.sidebar.header(
    "⚙️ Settings")

st.sidebar.number_input(
    label="Temperature",
    value=0.75,
    min_value=0.0, max_value=1.0,
    key="temperature",
    step=0.05,
    help="The level of creativity in the responses.\n- Higher values mean more creative responses.\n- Lower values mean more focused and deterministic responses.",
)


st.sidebar.selectbox(
    options=st.session_state.providers,
    label="Select Provider:",
    index=None,
    key="provider",
    help="Choose the AI provider for your model. You can also use Ollama for local models.",
)

st.sidebar.selectbox(
    label="Select Model:",
    options=get_model_list(
        st.session_state.provider),
    index=None,
    placeholder="Choose Model" if st.session_state.provider else "Choose provider first",
    key="model",
    help="Select the model you want to use for generating responses.\n- Make sure the provider is set first.\n- Selected model must support tool usage.",
)

st.sidebar.text_input(
    label="Enter your API Key:",
    placeholder="API Key 👀",
    type="password",
    key="user_api_key",
    value=st.session_state.user_api_key if st.session_state.user_api_key else None,
    help="(Optional) API Key in case you have selected provider other than Ollama"
)


# ----------------------------------------------------------------------------------------
# LangChain Components:
# ----------------------------------------------------------------------------------------

# Wikipedia Tool:
wrapper_wiki = WikipediaAPIWrapper(
    wiki_client=None, top_k_results=3,
    doc_content_chars_max=300
)
tool_wiki = WikipediaQueryRun(api_wrapper=wrapper_wiki)


# arXiv tool:
wrapper_arxiv = ArxivAPIWrapper(
    arxiv_search=None, arxiv_exceptions=None,
    top_k_results=3, doc_content_chars_max=300
)
tool_arxiv = ArxivQueryRun(api_wrapper=wrapper_arxiv)


# DuckDuckGo Search Tool:
web_search = DuckDuckGoSearchRun(name="Web Search")


# All tools:
tools = [web_search, tool_arxiv, tool_wiki]


# ----------------------------------------------------------------------------------------
# Main Content:
# ----------------------------------------------------------------------------------------

# First get LLM instance:
if not st.session_state.provider or not st.session_state.model:
    st.warning("Please select a provider and model from the sidebar to start using the search engine.")
    st.stop()
else:
    llm = get_llm_instance(
        provider=st.session_state.provider,
        model=st.session_state.model,
        temperature=st.session_state.temperature
    )


# Render entire chat history:
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        st.chat_message("user").write(message.content)
    elif isinstance(message, AIMessage):
        st.chat_message("assistant").write(message.content)
    elif isinstance(message, ToolMessage):
        st.chat_message("tool", avatar="⚙️").write(message.content)
    else:
        st.chat_message("system").write(type(message) + message.content)


if prompt := st.chat_input("Ask me anything..."):
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    st.chat_message("Human").write(prompt)

    search_agent = initialize_agent(
        tools=tools, llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handling_parsing_errors=True
    )

    with st.container(border=True):
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = search_agent.run(st.session_state.chat_history, callbacks=[st_cb])
            st.write(response)
            st.session_state.chat_history.append(AIMessage(content=response))
