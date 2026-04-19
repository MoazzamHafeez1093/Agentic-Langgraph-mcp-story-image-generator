"""
PROJECT MONTAGE – Phase 1 & 2
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
#  Output Paths – Phase 1
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
IMAGE_ASSETS_DIR = OUTPUT_DIR / "image_assets"
SCENE_MANIFEST_PATH = OUTPUT_DIR / "scene_manifest.json"
CHARACTER_DB_PATH = OUTPUT_DIR / "character_db.json"

# ChromaDB persistence directory
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# ─────────────────────────────────────────────
#  Output Paths – Phase 2: The Studio Floor
# ─────────────────────────────────────────────
RAW_SCENES_DIR = OUTPUT_DIR / "raw_scenes"
AUDIO_DIR = OUTPUT_DIR / "audio"
FRAMES_DIR = OUTPUT_DIR / "frames"
TASK_GRAPH_LOG_PATH = OUTPUT_DIR / "task_graph_log.json"
PHASE2_CHECKPOINT_PATH = OUTPUT_DIR / "phase2_checkpoint.json"

# ─────────────────────────────────────────────
#  Voice Mapping – Character → Microsoft Neural Voice
#  Each character gets a unique, distinct voice
# ─────────────────────────────────────────────
DEFAULT_VOICE_MAP = {
    "default_male_1": "en-US-GuyNeural",
    "default_male_2": "en-US-AndrewNeural",
    "default_male_3": "en-GB-RyanNeural",
    "default_female_1": "en-US-JennyNeural",
    "default_female_2": "en-US-AriaNeural",
    "default_female_3": "en-GB-SoniaNeural",
    "default_neutral_1": "en-US-AndrewNeural",
    "default_neutral_2": "en-US-JennyNeural",
}

# Pool of voices to auto-assign to characters
VOICE_POOL = [
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-US-AndrewNeural",
    "en-US-AriaNeural",
    "en-GB-RyanNeural",
    "en-GB-SoniaNeural",
    "en-AU-WilliamNeural",
    "en-AU-NatashaNeural",
]

# ─────────────────────────────────────────────
#  Ensure required directories exist
# ─────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)
IMAGE_ASSETS_DIR.mkdir(exist_ok=True)
CHROMA_DB_DIR.mkdir(exist_ok=True)
RAW_SCENES_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)
