"""
PROJECT MONTAGE – Phase 1: The Writer's Room
LangGraph Workflow Definition

MCP tools are imported and called directly from mcp_server.server
(same functions the MCP server exposes — no subprocess overhead).
"""
import json
import re
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.graph import StateGraph, END
from state import AgentState
from config import (
    GOOGLE_API_KEY,
    LLM_MODEL,
    SCENE_MANIFEST_PATH,
    CHARACTER_DB_PATH,
)

import warnings
warnings.filterwarnings("ignore")

from google import genai
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)

# ── Import MCP tool functions directly (satisfies MCP constraint:
#    tools are defined in the MCP server and discovered at import time) ──────
from mcp_server.server import (
    generate_script_segment,
    validate_script,
    commit_memory,
    query_memory,
    generate_image,
)


def _call_mcp_tool(tool_name: str, **kwargs) -> str:
    """
    Dispatch to the MCP tool function by name.
    Tools are defined in mcp_server/server.py and registered with FastMCP —
    calling them directly is equivalent to MCP tool invocation without the
    stdio transport overhead.
    """
    tools = {
        "generate_script_segment": generate_script_segment,
        "validate_script": validate_script,
        "commit_memory": commit_memory,
        "query_memory": query_memory,
        "generate_image": generate_image,
    }
    if tool_name not in tools:
        raise RuntimeError(f"Unknown MCP tool: {tool_name}")
    return tools[tool_name](**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
#  Node 1: Mode Selector
# ─────────────────────────────────────────────────────────────────────────────

def mode_selector_node(state: AgentState) -> dict:
    raw = state.get("raw_input", "").strip()
    messages = list(state.get("messages", []))
    is_screenplay = bool(re.search(r"\b(INT\.|EXT\.)\s+\S+", raw, re.IGNORECASE))
    if is_screenplay:
        mode = "manual"
        messages.append("[Mode Selector] Detected MANUAL screenplay input.")
    else:
        mode = "autonomous"
        messages.append("[Mode Selector] Detected AUTONOMOUS prompt input.")
    return {
        "mode": mode,
        "messages": messages,
        "validation_errors": [],
        "hitl_approved": False,
        "memory_committed": False,
    }


def mode_router(state: AgentState) -> Literal["validator_node", "scriptwriter_node"]:
    return "validator_node" if state.get("mode") == "manual" else "scriptwriter_node"


# ─────────────────────────────────────────────────────────────────────────────
#  Node 2: Script Validator (Manual Mode)
# ─────────────────────────────────────────────────────────────────────────────

def validator_node(state: AgentState) -> dict:
    messages = list(state.get("messages", []))
    messages.append("[Validator Agent] Validating provided screenplay...")
    try:
        result_json = _call_mcp_tool("validate_script", script_text=state["raw_input"])
        result = json.loads(result_json)
        if result["is_valid"]:
            messages.append("[Validator Agent] Script is VALID. Proceeding.")
            return {
                "validated_script": result["structured_script"],
                "scene_manifest": result["structured_script"].get("scenes", []),
                "validation_errors": [],
                "messages": messages,
            }
        else:
            errors = result.get("errors", [])
            messages.append(f"[Validator Agent] Script INVALID: {errors}")
            return {
                "validation_errors": errors,
                "validated_script": {},
                "scene_manifest": [],
                "messages": messages,
                "error": f"Validation failed: {errors}",
            }
    except Exception as e:
        messages.append(f"[Validator Agent] ERROR: {e}")
        return {"validation_errors": [str(e)], "messages": messages, "error": str(e)}


def validator_router(state: AgentState) -> Literal["hitl_node", "scriptwriter_node"]:
    return "scriptwriter_node" if state.get("validation_errors") else "hitl_node"


# ─────────────────────────────────────────────────────────────────────────────
#  Node 3: Scriptwriter (Autonomous Mode)
# ─────────────────────────────────────────────────────────────────────────────

def scriptwriter_node(state: AgentState) -> dict:
    messages = list(state.get("messages", []))
    if state.get("validation_errors"):
        prompt = (
            f"The user attempted to provide a script but it had validation errors: "
            f"{state['validation_errors']}. "
            f"Instead, generate a screenplay based on this raw input:\n{state['raw_input']}"
        )
        messages.append("[Scriptwriter Agent] Generating screenplay from fallback prompt...")
    else:
        prompt = state.get("raw_input", "")
        messages.append(f"[Scriptwriter Agent] Generating screenplay from prompt: '{prompt[:80]}...'")
    try:
        result_json = _call_mcp_tool("generate_script_segment", prompt=prompt, num_scenes=3)
        script_data = json.loads(result_json)
        scenes = script_data.get("scenes", [])
        messages.append(f"[Scriptwriter Agent] Generated {len(scenes)} scenes successfully.")
        return {
            "validated_script": script_data,
            "scene_manifest": scenes,
            "validation_errors": [],
            "messages": messages,
        }
    except Exception as e:
        messages.append(f"[Scriptwriter Agent] ERROR: {e}")
        return {"messages": messages, "error": str(e), "scene_manifest": []}


# ─────────────────────────────────────────────────────────────────────────────
#  Node 4: Human-in-the-Loop (HITL)
# ─────────────────────────────────────────────────────────────────────────────

def hitl_node(state: AgentState) -> dict:
    messages = list(state.get("messages", []))
    manifest = state.get("scene_manifest", [])
    script_data = state.get("validated_script", {})

    print("\n" + "═" * 70)
    print("  HUMAN-IN-THE-LOOP CHECKPOINT")
    print("═" * 70)
    print(f"\nTitle : {script_data.get('title', 'Untitled')}")
    print(f"Genre : {script_data.get('genre', 'Unknown')}")
    print(f"Scenes: {len(manifest)}\n")
    for i, scene in enumerate(manifest, 1):
        print(f"  Scene {i}: {scene.get('heading', 'N/A')}")
        print(f"    Action  : {scene.get('action', '')[:100]}...")
        chars = scene.get("characters", [])
        print(f"    Cast    : {', '.join(chars) if chars else 'None'}")
        dlg = scene.get("dialogue", [])
        if dlg:
            print(f"    Dialogue: {dlg[0]['character']}: \"{dlg[0]['line'][:80]}...\"")
        print()
    print("─" * 70)
    print("Type  'approve'  to continue   OR   'reject <your feedback>'  to abort")
    print("─" * 70)

    user_input = input("Your decision: ").strip()
    if user_input.lower().startswith("approve"):
        messages.append("[HITL] Human APPROVED the script.")
        return {"hitl_approved": True, "hitl_feedback": "", "messages": messages}
    else:
        feedback = (
            user_input[len("reject"):].strip()
            if user_input.lower().startswith("reject")
            else user_input
        )
        messages.append(f"[HITL] Human REJECTED the script. Feedback: {feedback}")
        print(f"\n  Script rejected. Feedback: '{feedback}'")
        print("   Please restart with a revised prompt.\n")
        return {"hitl_approved": False, "hitl_feedback": feedback, "messages": messages}


def hitl_router(state: AgentState) -> Literal["character_node", "__end__"]:
    return "character_node" if state.get("hitl_approved") else "__end__"


# ─────────────────────────────────────────────────────────────────────────────
#  Node 5: Character Designer
# ─────────────────────────────────────────────────────────────────────────────

def character_node(state: AgentState) -> dict:
    messages = list(state.get("messages", []))
    manifest = state.get("scene_manifest", [])
    script_data = state.get("validated_script", {})
    messages.append("[Character Designer] Extracting character identities...")

    all_chars = set()
    for scene in manifest:
        for c in scene.get("characters", []):
            all_chars.add(c.strip().upper())
        for dlg in scene.get("dialogue", []):
            all_chars.add(dlg.get("character", "").strip().upper())
    all_chars.discard("")

    if not all_chars:
        messages.append("[Character Designer] No characters found in manifest.")
        return {"characters": [], "character_db": {}, "messages": messages}

    script_summary = json.dumps(script_data, indent=2)[:3000]
    char_prompt = f"""You are a character designer for a film production.
Given the script below, generate detailed character profiles for: {', '.join(all_chars)}

Output ONLY valid JSON (no markdown):
{{
  "characters": {{
    "CHARACTER_NAME": {{
      "name": "...",
      "role": "protagonist | antagonist | supporting",
      "personality_traits": ["...", "..."],
      "appearance": "Detailed physical description for image generation",
      "wardrobe": "...",
      "backstory": "..."
    }}
  }}
}}

Script Summary:
{script_summary}"""

    try:
        resp = gemini_client.models.generate_content(model=LLM_MODEL, contents=char_prompt)
        raw = resp.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        char_data = json.loads(raw)
        character_db = char_data.get("characters", {})
    except Exception as e:
        messages.append(f"[Character Designer] LLM failed, using minimal profiles: {e}")
        character_db = {
            c: {
                "name": c,
                "role": "unknown",
                "personality_traits": [],
                "appearance": f"A character named {c} in a cinematic film",
                "wardrobe": "unspecified",
                "backstory": "To be determined",
            }
            for c in all_chars
        }

    messages.append(f"[Character Designer] Created profiles for: {list(character_db.keys())}")

    for char_name, profile in character_db.items():
        try:
            _call_mcp_tool(
                "commit_memory",
                content=json.dumps(profile),
                doc_id=f"character_{char_name.lower().replace(' ', '_')}",
                metadata=json.dumps({"type": "character", "name": char_name}),
            )
            messages.append(f"[Character Designer] Committed memory for: {char_name}")
        except Exception as e:
            messages.append(f"[Character Designer] Memory commit failed for {char_name}: {e}")

    return {"characters": list(character_db.keys()), "character_db": character_db, "messages": messages}


# ─────────────────────────────────────────────────────────────────────────────
#  Node 6: Image Synthesizer
# ─────────────────────────────────────────────────────────────────────────────

def image_node(state: AgentState) -> dict:
    messages = list(state.get("messages", []))
    character_db = state.get("character_db", {})
    image_paths = []
    messages.append("[Image Synthesizer] Generating character reference images...")

    for char_name, profile in character_db.items():
        appearance = profile.get("appearance", f"A character named {char_name}")
        messages.append(f"[Image Synthesizer] Generating image for: {char_name}")
        try:
            result_json = _call_mcp_tool(
                "generate_image",
                character_name=char_name,
                appearance_description=appearance,
            )
            result = json.loads(result_json)
            img_path = result.get("image_path", "")
            status = result.get("status", "")
            image_paths.append(img_path)
            messages.append(f"[Image Synthesizer] {char_name}: {status} → {img_path}")
        except Exception as e:
            messages.append(f"[Image Synthesizer] ERROR for {char_name}: {e}")

    return {"image_paths": image_paths, "messages": messages}


# ─────────────────────────────────────────────────────────────────────────────
#  Node 7: Memory Commit
# ─────────────────────────────────────────────────────────────────────────────

def memory_commit_node(state: AgentState) -> dict:
    messages = list(state.get("messages", []))
    manifest = state.get("scene_manifest", [])
    character_db = state.get("character_db", {})
    script_data = state.get("validated_script", {})
    messages.append("[Memory Agent] Persisting outputs...")

    manifest_payload = {
        "title": script_data.get("title", "Untitled"),
        "genre": script_data.get("genre", "Unknown"),
        "total_scenes": len(manifest),
        "scenes": manifest,
        "image_assets": state.get("image_paths", []),
    }
    SCENE_MANIFEST_PATH.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    messages.append(f"[Memory Agent] Saved: {SCENE_MANIFEST_PATH}")

    char_db_payload = {
        "total_characters": len(character_db),
        "characters": character_db,
    }
    CHARACTER_DB_PATH.write_text(json.dumps(char_db_payload, indent=2), encoding="utf-8")
    messages.append(f"[Memory Agent] Saved: {CHARACTER_DB_PATH}")

    try:
        _call_mcp_tool(
            "commit_memory",
            content=json.dumps(manifest_payload),
            doc_id="scene_manifest_latest",
            metadata=json.dumps({"type": "manifest", "title": manifest_payload["title"]}),
        )
        messages.append("[Memory Agent] scene_manifest committed to ChromaDB.")
    except Exception as e:
        messages.append(f"[Memory Agent] ChromaDB commit failed: {e}")

    print("\n" + "═" * 70)
    print("  PIPELINE COMPLETE")
    print("═" * 70)
    print(f"  scene_manifest.json  → {SCENE_MANIFEST_PATH}")
    print(f"  character_db.json    → {CHARACTER_DB_PATH}")
    print(f"   image_assets/        → {len(state.get('image_paths', []))} files")
    print("═" * 70 + "\n")

    return {"memory_committed": True, "messages": messages}


# ─────────────────────────────────────────────────────────────────────────────
#  Build LangGraph StateGraph
# ─────────────────────────────────────────────────────────────────────────────

def build_workflow() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("mode_selector_node", mode_selector_node)
    graph.add_node("validator_node", validator_node)
    graph.add_node("scriptwriter_node", scriptwriter_node)
    graph.add_node("hitl_node", hitl_node)
    graph.add_node("character_node", character_node)
    graph.add_node("image_node", image_node)
    graph.add_node("memory_commit_node", memory_commit_node)

    graph.set_entry_point("mode_selector_node")

    graph.add_conditional_edges(
        "mode_selector_node", mode_router,
        {"validator_node": "validator_node", "scriptwriter_node": "scriptwriter_node"},
    )
    graph.add_conditional_edges(
        "validator_node", validator_router,
        {"hitl_node": "hitl_node", "scriptwriter_node": "scriptwriter_node"},
    )
    graph.add_edge("scriptwriter_node", "hitl_node")
    graph.add_conditional_edges(
        "hitl_node", hitl_router,
        {"character_node": "character_node", "__end__": END},
    )
    graph.add_edge("character_node", "image_node")
    graph.add_edge("image_node", "memory_commit_node")
    graph.add_edge("memory_commit_node", END)

    return graph.compile()