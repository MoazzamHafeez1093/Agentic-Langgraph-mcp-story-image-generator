"""
PROJECT MONTAGE – Phase 1: The Writer's Room
Configuration Module
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
#  API Keys
# ─────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ─────────────────────────────────────────────
#  LLM Settings
# ─────────────────────────────────────────────
LLM_MODEL = "gemini-2.5-flash"
IMAGE_MODEL = "gemini-2.5-flash"

# ─────────────────────────────────────────────
#  MCP Server Settings
# ─────────────────────────────────────────────
MCP_SERVER_HOST = "localhost"
MCP_SERVER_PORT = 8765
MCP_SERVER_URL = f"http://{MCP_SERVER_HOST}:{MCP_SERVER_PORT}"

# ─────────────────────────────────────────────
#  Output Paths
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
IMAGE_ASSETS_DIR = OUTPUT_DIR / "image_assets"
SCENE_MANIFEST_PATH = OUTPUT_DIR / "scene_manifest.json"
CHARACTER_DB_PATH = OUTPUT_DIR / "character_db.json"

# ChromaDB persistence directory
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# Ensure required directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
IMAGE_ASSETS_DIR.mkdir(exist_ok=True)
CHROMA_DB_DIR.mkdir(exist_ok=True)
