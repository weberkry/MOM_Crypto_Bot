import os
from typing import TypedDict, Optional
from dotenv import load_dotenv

from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI 
from langchain_core.messages import HumanMessage, AIMessage

from langchain_community.vectorstores import FAISS
#news
from newsapi import NewsApiClient
import praw

# Tools (custom toolbox)
from mom.toolbox import toolbox, rag_risk   # ✅ Import rag_risk explicitly

# Load environment
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))


# ---- Agent State ----
class AgentState(TypedDict, total=False):
    input: str
    translated_input: Optional[str]
    detected_lang: Optional[str]
    messages: list
    agent_output: Optional[str]
    final_output: Optional[str]


# ---- LLM ----
llm = ChatOpenAI(
    api_key=OPENAI_KEY,
    model=OPENAI_MODEL,
    temperature=OPENAI_TEMPERATURE,
)


# ---- Translation ----
def translate_in_node(state: AgentState):
    from deep_translator import GoogleTranslator
    try:
        from langdetect import detect
        detected_lang = detect(state["input"])
    except Exception:
        detected_lang = "en"

    if detected_lang != "en":
        translated = GoogleTranslator(source=detected_lang, target="en").translate(state["input"])
    else:
        translated = state["input"]

    return {
        "detected_lang": detected_lang,
        "translated_input": translated,
        "messages": [HumanMessage(content=translated)],
    }


def translate_back_node(state: AgentState):
    from deep_translator import GoogleTranslator
    output = state.get("agent_output", "")
    if state.get("detected_lang") and state["detected_lang"] != "en":
        translated = GoogleTranslator(source="en", target=state["detected_lang"]).translate(output)
    else:
        translated = output
    return {"final_output": translated}


# ---- Reasoning / Tool Nodes ----
tool_node = ToolNode(tools=toolbox)


def llm_node(state: AgentState):
    messages = state.get("messages", [])
    if not messages:
        messages = [HumanMessage(content=state.get("translated_input", state.get("input", "")))]
    result = llm.invoke(messages)
    return {
        "agent_output": result.content,
        "messages": messages + [AIMessage(content=result.content)],
    }


# ---- Build Graph ----
graph = StateGraph(AgentState)
graph.add_node("translate_in", translate_in_node)
graph.add_node("llm", llm_node)
graph.add_node("tools", tool_node)
graph.add_node("translate_back", translate_back_node)

graph.set_entry_point("translate_in")
graph.add_edge("translate_in", "llm")
graph.add_edge("llm", "tools")
graph.add_edge("tools", "translate_back")
graph.set_finish_point("translate_back")

agent_app = graph.compile()


# ---- Wrappers ----
def run(user_input: str) -> str:
    """Generic free-form pipeline (LangGraph path)."""
    if not user_input.strip():
        return "No message provided."
    result = agent_app.invoke({"input": user_input})
    return result.get("final_output", result.get("agent_output", "No response generated"))


def run_risk(asset: str = "BTC") -> str:
    """Direct risk evaluation tool call (bypasses LLM)."""
    try:
        return rag_risk(asset)
    except Exception as e:
        return f"Error running rag_risk on {asset}: {e}"


# ---- Debug / Test ----
if __name__ == "__main__":
    print(">>> /risk BTC")
    print(run_risk("BTC"))

    print("\n>>> Freeform")
    print(run("What is the risk level for ETH right now based on news and Hurst?"))
