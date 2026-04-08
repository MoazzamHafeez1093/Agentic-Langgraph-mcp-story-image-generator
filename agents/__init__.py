"""
PROJECT MONTAGE – Phase 1: The Writer's Room
LangGraph Workflow – agents/*.py

Each agent function is a LangGraph node:
  - Receives AgentState
  - Calls MCP tools (via mcp_client)
  - Returns partial state dict
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from config import GOOGLE_API_KEY, LLM_MODEL

def _get_llm():
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7,
    )
