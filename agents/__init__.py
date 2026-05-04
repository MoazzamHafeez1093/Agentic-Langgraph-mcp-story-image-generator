"""
PROJECT MONTAGE – Phase 1: The Writer's Room
agents/__init__.py

NOTE: All agent logic lives in graph/workflow.py as LangGraph nodes.
      This package is kept for project structure clarity.
      The project uses google.genai SDK directly (not langchain_google_genai).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import GOOGLE_API_KEY, LLM_MODEL
from google import genai

def _get_gemini_client():
    """Return a configured Gemini client (google.genai SDK)."""
    return genai.Client(api_key=GOOGLE_API_KEY)
