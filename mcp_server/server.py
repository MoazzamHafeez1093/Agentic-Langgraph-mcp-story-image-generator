"""
PROJECT MONTAGE – Phase 1: The Writer's Room
MCP Server – Exposes all tools for dynamic agent discovery

Run with: python mcp_server/server.py
"""
import json
import re
import base64
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore")

# ── NEW SDK ──────────────────────────────────────────────────────────────
from google import genai
from google.genai import types
# ─────────────────────────────────────────────────────────────────────────

from fastmcp import FastMCP
import chromadb
from chromadb.utils import embedding_functions

from config import (
    GOOGLE_API_KEY,
    LLM_MODEL,
    IMAGE_MODEL,
    CHROMA_DB_DIR,
    IMAGE_ASSETS_DIR,
)

# ─────────────────────────────────────────────
#  Initialise Gemini client
# ─────────────────────────────────────────────
client = genai.Client(api_key=GOOGLE_API_KEY)

# ─────────────────────────────────────────────
#  ChromaDB – DefaultEmbeddingFunction only
#
#  SentenceTransformerEmbeddingFunction downloads
#  the model on first run (~400 MB) which causes
#  the subprocess to hit the 120s timeout.
#  DefaultEmbeddingFunction is instant and works
#  perfectly for this project's needs.
# ─────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
ef = embedding_functions.DefaultEmbeddingFunction()
memory_collection = chroma_client.get_or_create_collection(
    name="montage_memory",
    embedding_function=ef,
)

# ─────────────────────────────────────────────
#  FastMCP app
# ─────────────────────────────────────────────
mcp = FastMCP(
    name="project_montage_mcp",
    instructions=(
        "MCP server for PROJECT MONTAGE Phase 1. "
        "Provides tools: generate_script_segment, validate_script, "
        "commit_memory, query_memory, generate_image."
    ),
)


# ─────────────────────────────────────────────
#  Tool 1: generate_script_segment
# ─────────────────────────────────────────────
@mcp.tool()
def generate_script_segment(prompt: str, num_scenes: int = 3) -> str:
    """
    Generate a structured multi-scene screenplay from a user prompt.

    Args:
        prompt: The creative brief or story idea.
        num_scenes: Number of scenes to generate (default 3).

    Returns:
        JSON string with a list of scene objects, each containing:
        scene_id, heading, action, characters, dialogue (list), visual_cues.
    """
    system_prompt = f"""You are an expert Hollywood screenwriter.
Given the user's creative prompt, generate exactly {num_scenes} scenes.

Output ONLY valid JSON in this exact format (no markdown, no explanation):
{{
  "title": "Story Title",
  "genre": "Genre",
  "scenes": [
    {{
      "scene_id": 1,
      "heading": "INT. LOCATION - DAY",
      "action": "Scene action description here.",
      "characters": ["CHARACTER_A", "CHARACTER_B"],
      "dialogue": [
        {{"character": "CHARACTER_A", "line": "Dialogue line."}},
        {{"character": "CHARACTER_B", "line": "Response line."}}
      ],
      "visual_cues": ["Close-up on CHARACTER_A's face", "Warm golden lighting"]
    }}
  ]
}}

User Prompt: {prompt}"""

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=system_prompt,
    )
    raw = response.text.strip()

    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    # Validate it's real JSON before returning
    json.loads(raw)
    return raw


# ─────────────────────────────────────────────
#  Tool 2: validate_script
# ─────────────────────────────────────────────
@mcp.tool()
def validate_script(script_text: str) -> str:
    """
    Validate a manually provided screenplay for correct structure.

    Args:
        script_text: Raw script text provided by the user.

    Returns:
        JSON string with keys:
          - is_valid (bool)
          - errors (list of strings)
          - structured_script (dict | null)
    """
    errors = []

    has_headings = bool(re.search(r"\b(INT\.|EXT\.)\s+\S+", script_text, re.IGNORECASE))
    if not has_headings:
        errors.append("Missing scene headings (INT./EXT. LOCATION - TIME)")

    has_dialogue = bool(re.search(r"^[A-Z][A-Z\s]+$", script_text, re.MULTILINE))
    if not has_dialogue:
        errors.append("Missing dialogue labels (CHARACTER NAME in ALL CAPS on its own line)")

    lines = [l.strip() for l in script_text.split("\n") if l.strip()]
    action_lines = [
        l for l in lines
        if not re.match(r"^(INT\.|EXT\.)", l, re.IGNORECASE)
        and not re.match(r"^[A-Z][A-Z\s]+$", l)
        and not l.startswith("(")
    ]
    if len(action_lines) < 2:
        errors.append("Insufficient action descriptions")

    if errors:
        return json.dumps({"is_valid": False, "errors": errors, "structured_script": None})

    parse_prompt = f"""Convert this screenplay into structured JSON.
Output ONLY valid JSON (no markdown):
{{
  "title": "...",
  "genre": "...",
  "scenes": [
    {{
      "scene_id": 1,
      "heading": "...",
      "action": "...",
      "characters": [...],
      "dialogue": [{{"character": "...", "line": "..."}}],
      "visual_cues": [...]
    }}
  ]
}}

Script:
{script_text}"""

    resp = client.models.generate_content(
        model=LLM_MODEL,
        contents=parse_prompt,
    )
    raw = resp.text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        structured = json.loads(raw)
        return json.dumps({"is_valid": True, "errors": [], "structured_script": structured})
    except json.JSONDecodeError:
        return json.dumps({
            "is_valid": False,
            "errors": ["LLM returned invalid JSON during parsing"],
            "structured_script": None,
        })


# ─────────────────────────────────────────────
#  Tool 3: commit_memory
# ─────────────────────────────────────────────
@mcp.tool()
def commit_memory(content: str, doc_id: str, metadata: str = "{}") -> str:
    """
    Store text with optional metadata in the ChromaDB vector store.

    Args:
        content: Text content to embed and store.
        doc_id: Unique identifier for this document.
        metadata: JSON string of metadata key-value pairs.

    Returns:
        Confirmation string.
    """
    try:
        meta_dict = json.loads(metadata)
    except Exception:
        meta_dict = {}

    memory_collection.upsert(
        documents=[content],
        ids=[doc_id],
        metadatas=[meta_dict],
    )
    return f"Memory committed: id={doc_id}"


# ─────────────────────────────────────────────
#  Tool 4: query_memory
# ─────────────────────────────────────────────
@mcp.tool()
def query_memory(query: str, n_results: int = 3) -> str:
    """
    Retrieve semantically similar documents from ChromaDB.

    Args:
        query: Search query text.
        n_results: Number of results to return (default 3).

    Returns:
        JSON string with list of {id, document, metadata, distance}.
    """
    count = memory_collection.count()
    if count == 0:
        return json.dumps([])

    n_results = min(n_results, count)
    results = memory_collection.query(
        query_texts=[query],
        n_results=n_results,
    )

    output = []
    for i, doc_id in enumerate(results["ids"][0]):
        output.append({
            "id": doc_id,
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return json.dumps(output, indent=2)


# ─────────────────────────────────────────────
#  Tool 5: generate_image
# ─────────────────────────────────────────────
@mcp.tool()
def generate_image(character_name: str, appearance_description: str) -> str:
    """
    Generate a character reference image using Google Gemini image generation.

    Args:
        character_name: Name of the character (used for filename).
        appearance_description: Detailed visual description for image generation.

    Returns:
        JSON string with keys: character_name, image_path, status.
    """
    safe_name = re.sub(r"[^\w\-]", "_", character_name.lower())
    prompt = (
        f"Character reference illustration for '{character_name}'. "
        f"{appearance_description}. "
        "Cinematic lighting, high detail, character portrait, film production style."
    )

    # ── Try Imagen 3 first (requires paid tier) ───────────────────────────
    imagen_err = None
    try:
        response = client.models.generate_images(
            model="imagen-3.0-generate-002",
            prompt=prompt,
            config=types.GenerateImagesConfig(number_of_images=1),
        )
        image_bytes = response.generated_images[0].image.image_bytes
        image_path = IMAGE_ASSETS_DIR / f"{safe_name}.png"
        image_path.write_bytes(image_bytes)
        return json.dumps({
            "character_name": character_name,
            "image_path": str(image_path),
            "status": "success",
        })
    except Exception as e:
        imagen_err = e

    # ── Fallback: gemini-2.0-flash image generation ───────────────────────
    try:
        flash_response = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )
        image_data = None
        for part in flash_response.candidates[0].content.parts:
            if part.inline_data is not None:
                image_data = base64.b64decode(part.inline_data.data)
                break

        if image_data:
            image_path = IMAGE_ASSETS_DIR / f"{safe_name}.png"
            image_path.write_bytes(image_data)
            return json.dumps({
                "character_name": character_name,
                "image_path": str(image_path),
                "status": "success_flash_fallback",
            })
    except Exception:
        pass

    # ── Final fallback: placeholder file ─────────────────────────────────
    placeholder_path = IMAGE_ASSETS_DIR / f"{safe_name}_placeholder.txt"
    placeholder_path.write_text(
        f"Image generation placeholder for {character_name}\n"
        f"Description: {appearance_description}\n"
        f"Imagen error: {imagen_err}"
    )
    return json.dumps({
        "character_name": character_name,
        "image_path": str(placeholder_path),
        "status": "placeholder_created",
    })


# ─────────────────────────────────────────────
#  Entrypoint (STDIO transport for LangGraph)
#
#  IMPORTANT: do NOT print() anything to stdout
#  here. stdout is the MCP JSON-RPC channel —
#  any extra bytes will corrupt the stream.
# ─────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run(transport="stdio")