"""
PROJECT MONTAGE – Phase 1 & Phase 2
LangGraph Workflow Definition

Phase 1: The Writer's Room  (Nodes 1-7)
Phase 2: The Studio Floor   (Nodes 8-13, Parallel Send() Architecture)

MCP tools are imported and called directly from mcp_server.server
(same functions the MCP server exposes — no subprocess overhead).
"""
import json
import re
import sys
import time
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.graph import StateGraph, END
from langgraph.types import Send
from state import AgentState
from config import (
    GOOGLE_API_KEY,
    LLM_MODEL,
    SCENE_MANIFEST_PATH,
    CHARACTER_DB_PATH,
    RAW_SCENES_DIR,
    AUDIO_DIR,
    TASK_GRAPH_LOG_PATH,
    PHASE2_CHECKPOINT_PATH,
    IMAGE_ASSETS_DIR,
)

import warnings
warnings.filterwarnings("ignore")

from google import genai
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)

# ── Import MCP tool functions directly (satisfies MCP constraint:
#    tools are defined in the MCP server and discovered at import time) ──────
from mcp_server.server import (
    # Phase 1 tools
    generate_script_segment,
    validate_script,
    commit_memory,
    query_memory,
    generate_image,
    # Phase 2 tools
    get_task_graph,
    voice_cloning_synthesizer,
    query_stock_footage,
    face_swapper,
    identity_validator,
    lip_sync_aligner,
)


def _call_mcp_tool(tool_name: str, **kwargs) -> str:
    """
    Dispatch to the MCP tool function by name.
    Tools are defined in mcp_server/server.py and registered with FastMCP —
    calling them directly is equivalent to MCP tool invocation without the
    stdio transport overhead.
    """
    tools = {
        # Phase 1
        "generate_script_segment": generate_script_segment,
        "validate_script": validate_script,
        "commit_memory": commit_memory,
        "query_memory": query_memory,
        "generate_image": generate_image,
        # Phase 2
        "get_task_graph": get_task_graph,
        "voice_cloning_synthesizer": voice_cloning_synthesizer,
        "query_stock_footage": query_stock_footage,
        "face_swapper": face_swapper,
        "identity_validator": identity_validator,
        "lip_sync_aligner": lip_sync_aligner,
    }
    if tool_name not in tools:
        raise RuntimeError(f"Unknown MCP tool: {tool_name}")
    return tools[tool_name](**kwargs)


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 1 NODES  –  The Writer's Room
# ═══════════════════════════════════════════════════════════════════════


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

    print("\n" + "=" * 70)
    print("  HUMAN-IN-THE-LOOP CHECKPOINT")
    print("=" * 70)
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
    print("-" * 70)
    print("Type  'approve'  to continue   OR   'reject <your feedback>'  to abort")
    print("-" * 70)

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

    print("\n" + "=" * 70)
    print("  PHASE 1 COMPLETE")
    print("=" * 70)
    print(f"  scene_manifest.json  → {SCENE_MANIFEST_PATH}")
    print(f"  character_db.json    → {CHARACTER_DB_PATH}")
    print(f"   image_assets/        → {len(state.get('image_paths', []))} files")
    print("═" * 70 + "\n")

    return {"memory_committed": True, "messages": messages}


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 2 NODES  –  The Studio Floor: Video & Audio Synthesis
# ═══════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
#  Node 8: Scene Parser Agent
# ─────────────────────────────────────────────────────────────────────────────

def scene_parser_node(state: AgentState) -> dict:
    """
    Transform scene_manifest.json into executable tasks.
    Decompose scenes into a task graph with audio and video branches.
    Uses MCP tools: get_task_graph, commit_memory
    """
    messages = list(state.get("messages", []))
    messages.append("\n" + "=" * 70)
    messages.append("  PHASE 2: THE STUDIO FLOOR — Video & Audio Synthesis")
    messages.append("=" * 70)
    messages.append("[Scene Parser] Loading scene manifest and building task graph...")

    print("\n" + "=" * 70)
    print("  PHASE 2: THE STUDIO FLOOR — Video & Audio Synthesis")
    print("=" * 70)

    # Load scene manifest
    manifest_data = {}
    if SCENE_MANIFEST_PATH.exists():
        manifest_data = json.loads(SCENE_MANIFEST_PATH.read_text(encoding="utf-8"))
        messages.append(f"[Scene Parser] Loaded manifest: {manifest_data.get('title', 'Untitled')}")
        messages.append(f"[Scene Parser] Total scenes: {manifest_data.get('total_scenes', 0)}")
    else:
        # Use from state
        manifest_data = {
            "title": state.get("validated_script", {}).get("title", "Untitled"),
            "scenes": state.get("scene_manifest", []),
        }
        messages.append("[Scene Parser] Using manifest from pipeline state.")

    # Call MCP tool: get_task_graph
    task_graph_json = _call_mcp_tool(
        "get_task_graph",
        scene_manifest_json=json.dumps(manifest_data),
    )
    task_graph = json.loads(task_graph_json)

    messages.append(f"[Scene Parser] Task graph generated: {task_graph.get('total_tasks', 0)} tasks across {task_graph.get('total_scenes', 0)} scenes")
    messages.append(f"[Scene Parser] Parallel branches: {task_graph.get('parallel_branches', [])}")

    # Save task graph log
    TASK_GRAPH_LOG_PATH.write_text(json.dumps(task_graph, indent=2), encoding="utf-8")
    messages.append(f"[Scene Parser] Task graph log saved: {TASK_GRAPH_LOG_PATH}")

    # Commit task graph to memory for fault tolerance
    try:
        _call_mcp_tool(
            "commit_memory",
            content=json.dumps(task_graph),
            doc_id="task_graph_latest",
            metadata=json.dumps({"type": "task_graph", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}),
        )
        messages.append("[Scene Parser] Task graph committed to ChromaDB for recovery.")
    except Exception as e:
        messages.append(f"[Scene Parser] Memory commit failed: {e}")

    print(f"  Task Graph: {task_graph.get('total_tasks', 0)} tasks, {task_graph.get('total_scenes', 0)} scenes")
    print(f"  Branches: {task_graph.get('parallel_branches', [])}")

    return {
        "task_graph": task_graph.get("tasks", []),
        "messages": messages,
        "scene_manifest": manifest_data.get("scenes", state.get("scene_manifest", [])),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Parallel Router: Fan-out audio + video using Send() API
# ─────────────────────────────────────────────────────────────────────────────

def parallel_av_router(state: AgentState):
    """
    Uses LangGraph Send() API for parallel branching.
    For each scene, spawn both a voice_synth_node and a video_gen_node
    that will execute concurrently.
    """
    scenes = state.get("scene_manifest", [])
    character_db = state.get("character_db", {})
    image_paths = state.get("image_paths", [])
    sends = []

    for scene in scenes:
        scene_id = scene.get("scene_id", 1)

        # ── Audio Branch ──
        sends.append(Send("voice_synth_node", {
            "scene": scene,
            "scene_id": scene_id,
            "character_db": character_db,
        }))

        # ── Video Branch ──
        sends.append(Send("video_gen_node", {
            "scene": scene,
            "scene_id": scene_id,
            "image_paths": image_paths,
        }))

    return sends


# ─────────────────────────────────────────────────────────────────────────────
#  Node 9: Voice Synthesis Agent (runs in parallel per scene)
# ─────────────────────────────────────────────────────────────────────────────

def voice_synth_node(state: dict) -> dict:
    """
    Generate speech for a scene's dialogue using voice_cloning_synthesizer.
    Each character gets a unique neural voice for identity consistency.
    Runs in parallel with video_gen_node via Send() API.
    """
    scene = state.get("scene", {})
    scene_id = state.get("scene_id", 1)
    messages = []
    audio_results = []

    messages.append(f"[Voice Synth] 🎤 Processing Scene {scene_id}: {scene.get('heading', 'N/A')}")
    print(f"  🎤 Voice Synthesis: Scene {scene_id} — {scene.get('heading', '')}")

    dialogues = scene.get("dialogue", [])

    if not dialogues:
        messages.append(f"[Voice Synth] No dialogue in scene {scene_id}, generating narration...")
        # Generate narration from action description
        action = scene.get("action", "A cinematic scene unfolds.")
        try:
            result_json = _call_mcp_tool(
                "voice_cloning_synthesizer",
                character_name="NARRATOR",
                dialogue=action[:500],
                emotion="calm",
                scene_id=scene_id,
            )
            result = json.loads(result_json)
            audio_results.append(result)
            messages.append(f"[Voice Synth] Narration generated: {result.get('status')}")
        except Exception as e:
            messages.append(f"[Voice Synth] Narration failed: {e}")
    else:
        # Concatenate all dialogue for the scene
        all_dialogue = ""
        for dlg in dialogues:
            char = dlg.get("character", "UNKNOWN")
            line = dlg.get("line", "")
            # Clean parenthetical stage directions from the line
            clean_line = re.sub(r"\(.*?\)", "", line).strip()
            if clean_line:
                all_dialogue += f"{clean_line} "

        if all_dialogue.strip():
            # Detect emotion from scene context
            action = scene.get("action", "").lower()
            emotion = "neutral"
            if any(w in action for w in ["fear", "terror", "panic", "horror", "recoil"]):
                emotion = "fearful"
            elif any(w in action for w in ["anger", "furious", "rage"]):
                emotion = "angry"
            elif any(w in action for w in ["sad", "grief", "mourn", "cry"]):
                emotion = "sad"
            elif any(w in action for w in ["excit", "rush", "thrill"]):
                emotion = "excited"
            elif any(w in action for w in ["calm", "quiet", "serene", "still"]):
                emotion = "calm"

            # Use first character as primary voice for the scene
            primary_char = dialogues[0].get("character", "NARRATOR")
            try:
                result_json = _call_mcp_tool(
                    "voice_cloning_synthesizer",
                    character_name=primary_char,
                    dialogue=all_dialogue.strip(),
                    emotion=emotion,
                    scene_id=scene_id,
                )
                result = json.loads(result_json)
                audio_results.append(result)
                messages.append(f"[Voice Synth] Scene {scene_id} audio: {result.get('status')} ({result.get('voice_id')})")
            except Exception as e:
                messages.append(f"[Voice Synth] Scene {scene_id} audio failed: {e}")

    # Commit checkpoint
    try:
        _call_mcp_tool(
            "commit_memory",
            content=json.dumps({"scene_id": scene_id, "audio": audio_results}),
            doc_id=f"phase2_audio_scene_{scene_id}",
            metadata=json.dumps({"type": "phase2_audio", "scene_id": scene_id}),
        )
    except Exception:
        pass

    return {
        "audio_outputs": audio_results,
        "messages": messages,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Node 10: Video Generation Agent (runs in parallel per scene)
# ─────────────────────────────────────────────────────────────────────────────

def video_gen_node(state: dict) -> dict:
    """
    Generate scene visuals from character references and scene descriptions.
    Uses query_stock_footage MCP tool to create animated scene videos.
    Runs in parallel with voice_synth_node via Send() API.
    """
    scene = state.get("scene", {})
    scene_id = state.get("scene_id", 1)
    messages = []
    video_results = []

    messages.append(f"[Video Gen] 🎬 Processing Scene {scene_id}: {scene.get('heading', 'N/A')}")
    print(f"  🎬 Video Generation: Scene {scene_id} — {scene.get('heading', '')}")

    scene_desc = scene.get("action", "A cinematic scene")
    visual_cues = scene.get("visual_cues", [])

    try:
        result_json = _call_mcp_tool(
            "query_stock_footage",
            scene_description=scene_desc,
            visual_cues=json.dumps(visual_cues),
            scene_id=scene_id,
            duration=8.0,
        )
        result = json.loads(result_json)
        video_results.append(result)
        messages.append(f"[Video Gen] Scene {scene_id} video: {result.get('status')} ({result.get('frame_count', 0)} frames)")
    except Exception as e:
        messages.append(f"[Video Gen] Scene {scene_id} video failed: {e}")

    # Commit checkpoint
    try:
        _call_mcp_tool(
            "commit_memory",
            content=json.dumps({"scene_id": scene_id, "video": video_results}),
            doc_id=f"phase2_video_scene_{scene_id}",
            metadata=json.dumps({"type": "phase2_video", "scene_id": scene_id}),
        )
    except Exception:
        pass

    return {
        "video_outputs": video_results,
        "messages": messages,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Node 11: Face Swap Agent
# ─────────────────────────────────────────────────────────────────────────────

def face_swap_node(state: AgentState) -> dict:
    """
    Map generated characters onto video frames.
    MUST validate identity before mapping (uses identity_validator + face_swapper).
    """
    messages = list(state.get("messages", []))
    video_outputs = state.get("video_outputs", [])
    character_db = state.get("character_db", {})
    image_paths = state.get("image_paths", [])
    face_swap_results = []

    messages.append("[Face Swap] 🎭 Mapping character faces onto video frames...")
    print("  🎭 Face Swap: Mapping characters onto scenes...")

    for video_info in video_outputs:
        scene_id = video_info.get("scene_id", 1)
        video_path = video_info.get("mp4_path", "")

        if not video_path or video_info.get("status", "").startswith("error"):
            messages.append(f"[Face Swap] Skipping scene {scene_id}: no valid video")
            continue

        # Find the first valid character image for this scene
        char_image_path = ""
        char_name = ""

        for img_path in image_paths:
            if Path(img_path).exists() and Path(img_path).suffix.lower() in [".png", ".jpg", ".jpeg"]:
                char_image_path = img_path
                # Extract character name from filename
                char_name = Path(img_path).stem.upper().replace("_", " ")
                break

        if not char_image_path:
            messages.append(f"[Face Swap] No character image found for scene {scene_id}")
            face_swap_results.append({
                "scene_id": scene_id,
                "swapped_mp4_path": video_path,
                "status": "skipped: no character image",
            })
            continue

        # CRITICAL: Validate identity BEFORE face swap
        messages.append(f"[Face Swap] Validating identity for {char_name}...")
        try:
            validation_json = _call_mcp_tool(
                "identity_validator",
                character_name=char_name,
                character_image_path=char_image_path,
            )
            validation = json.loads(validation_json)

            if not validation.get("valid", False):
                messages.append(f"[Face Swap] ⚠️ Identity validation FAILED for {char_name}: {validation.get('details')}")
                face_swap_results.append({
                    "scene_id": scene_id,
                    "swapped_mp4_path": video_path,
                    "status": f"skipped: identity validation failed",
                })
                continue

            messages.append(f"[Face Swap] ✓ Identity validated for {char_name} (confidence: {validation.get('confidence')})")
        except Exception as e:
            messages.append(f"[Face Swap] Identity validation error: {e}")

        # Perform face swap
        try:
            swap_json = _call_mcp_tool(
                "face_swapper",
                video_path=video_path,
                character_image_path=char_image_path,
                character_name=char_name,
                scene_id=scene_id,
            )
            swap_result = json.loads(swap_json)
            face_swap_results.append({
                "scene_id": scene_id,
                "swapped_mp4_path": swap_result.get("swapped_video_path", video_path),
                "status": swap_result.get("status", "unknown"),
            })
            messages.append(f"[Face Swap] Scene {scene_id}: {swap_result.get('status')}")
        except Exception as e:
            messages.append(f"[Face Swap] Scene {scene_id} face swap error: {e}")
            face_swap_results.append({
                "scene_id": scene_id,
                "swapped_mp4_path": video_path,
                "status": f"error: {e}",
            })

    # Commit checkpoint
    try:
        _call_mcp_tool(
            "commit_memory",
            content=json.dumps(face_swap_results),
            doc_id="phase2_faceswap_checkpoint",
            metadata=json.dumps({"type": "phase2_faceswap"}),
        )
    except Exception:
        pass

    return {
        "face_swap_outputs": face_swap_results,
        "messages": messages,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Node 12: Lip Sync Agent
# ─────────────────────────────────────────────────────────────────────────────

def lip_sync_node(state: AgentState) -> dict:
    """
    Synchronize audio waveforms with facial movements.
    Frame-by-frame alignment ensuring speech timing = lip motion.
    Uses lip_sync_aligner MCP tool.
    """
    messages = list(state.get("messages", []))
    audio_outputs = state.get("audio_outputs", [])
    face_swap_outputs = state.get("face_swap_outputs", [])
    lip_sync_results = []

    messages.append("[Lip Sync] 👄 Synchronizing audio and video...")
    print("  👄 Lip Sync: Aligning audio with video frames...")

    # Build lookup maps
    audio_by_scene = {}
    for ao in audio_outputs:
        sid = ao.get("scene_id", 0)
        if ao.get("wav_path"):
            audio_by_scene[sid] = ao

    video_by_scene = {}
    for fs in face_swap_outputs:
        sid = fs.get("scene_id", 0)
        if fs.get("swapped_mp4_path"):
            video_by_scene[sid] = fs

    # Process each scene
    all_scene_ids = sorted(set(list(audio_by_scene.keys()) + list(video_by_scene.keys())))

    for scene_id in all_scene_ids:
        audio_info = audio_by_scene.get(scene_id, {})
        video_info = video_by_scene.get(scene_id, {})

        audio_path = audio_info.get("wav_path", "")
        video_path = video_info.get("swapped_mp4_path", "")

        if not video_path:
            messages.append(f"[Lip Sync] Skipping scene {scene_id}: no video available")
            continue

        messages.append(f"[Lip Sync] Aligning scene {scene_id}...")

        try:
            result_json = _call_mcp_tool(
                "lip_sync_aligner",
                audio_path=audio_path,
                video_path=video_path,
                scene_id=scene_id,
            )
            result = json.loads(result_json)
            lip_sync_results.append(result)
            messages.append(
                f"[Lip Sync] Scene {scene_id}: {result.get('status')} "
                f"(duration: {result.get('duration', 0):.1f}s)"
            )
        except Exception as e:
            messages.append(f"[Lip Sync] Scene {scene_id} alignment error: {e}")
            lip_sync_results.append({
                "final_mp4_path": video_path,
                "duration": 0,
                "scene_id": scene_id,
                "status": f"error: {e}",
            })

    # Commit checkpoint
    try:
        _call_mcp_tool(
            "commit_memory",
            content=json.dumps(lip_sync_results),
            doc_id="phase2_lipsync_checkpoint",
            metadata=json.dumps({"type": "phase2_lipsync"}),
        )
    except Exception:
        pass

    return {
        "lip_sync_outputs": lip_sync_results,
        "messages": messages,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Node 13: Phase 2 Output Node
# ─────────────────────────────────────────────────────────────────────────────

def phase2_output_node(state: AgentState) -> dict:
    """
    Final Phase 2 output — summarize all generated assets and save checkpoint.
    """
    messages = list(state.get("messages", []))
    lip_sync_outputs = state.get("lip_sync_outputs", [])
    audio_outputs = state.get("audio_outputs", [])
    video_outputs = state.get("video_outputs", [])
    task_graph = state.get("task_graph", [])

    messages.append("[Phase 2 Output] Finalizing all outputs...")

    # Build checkpoint for fault tolerance
    checkpoint = {
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_scenes": len(task_graph),
        "audio_generated": len(audio_outputs),
        "videos_generated": len(video_outputs),
        "lip_synced": len(lip_sync_outputs),
        "final_outputs": [],
    }

    for ls in lip_sync_outputs:
        if ls.get("final_mp4_path"):
            checkpoint["final_outputs"].append({
                "scene_id": ls.get("scene_id"),
                "mp4_path": ls.get("final_mp4_path"),
                "duration": ls.get("duration"),
                "status": ls.get("status"),
            })

    # Save checkpoint
    PHASE2_CHECKPOINT_PATH.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

    # Commit final state to memory
    try:
        _call_mcp_tool(
            "commit_memory",
            content=json.dumps(checkpoint),
            doc_id="phase2_final_checkpoint",
            metadata=json.dumps({"type": "phase2_complete", "timestamp": checkpoint["completed_at"]}),
        )
    except Exception:
        pass

    # Print summary
    print("\n" + "=" * 70)
    print("  PHASE 2 COMPLETE — THE STUDIO FLOOR")
    print("=" * 70)
    print(f"  Audio files generated : {len(audio_outputs)}")
    print(f"  Video files generated : {len(video_outputs)}")
    print(f"  Lip-synced outputs    : {len(lip_sync_outputs)}")
    print()
    print("  Final Outputs:")
    for output in checkpoint.get("final_outputs", []):
        print(f"    Scene {output['scene_id']}: {output['mp4_path']}")
        print(f"      Duration: {output['duration']:.1f}s | Status: {output['status']}")
    print()
    print(f"  Task Graph Log     → {TASK_GRAPH_LOG_PATH}")
    print(f"  Audio Directory    → {AUDIO_DIR}")
    print(f"  Raw Scenes         → {RAW_SCENES_DIR}")
    print(f"  Checkpoint         → {PHASE2_CHECKPOINT_PATH}")
    print("═" * 70 + "\n")

    messages.append("[Phase 2 Output] ✅ All assets generated successfully!")
    messages.append(f"[Phase 2 Output] Final videos: {len(checkpoint.get('final_outputs', []))}")

    return {
        "phase2_checkpoint": checkpoint,
        "messages": messages,
    }


# ═══════════════════════════════════════════════════════════════════════
#  BUILD LANGGRAPH WORKFLOWS
# ═══════════════════════════════════════════════════════════════════════


def build_workflow() -> StateGraph:
    """Build Phase 1 workflow: The Writer's Room."""
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


def build_phase2_workflow() -> StateGraph:
    """
    Build Phase 2 workflow: The Studio Floor.

    Uses Send() API for parallel branching:
      scene_parser → [Send(voice_synth), Send(video_gen)]  ← PARALLEL
                   → face_swap → lip_sync → output

    Nodes:
      - Scene_parser_node
      - Voice_synth_node    (parallel branch)
      - Video_gen_node      (parallel branch)
      - Face_swap_node
      - Lip_sync_node
      - Phase2_output_node
    """
    graph = StateGraph(AgentState)

    # Register all Phase 2 nodes
    graph.add_node("scene_parser_node", scene_parser_node)
    graph.add_node("voice_synth_node", voice_synth_node)
    graph.add_node("video_gen_node", video_gen_node)
    graph.add_node("face_swap_node", face_swap_node)
    graph.add_node("lip_sync_node", lip_sync_node)
    graph.add_node("phase2_output_node", phase2_output_node)

    # Entry point
    graph.set_entry_point("scene_parser_node")

    # ── PARALLEL BRANCHING via Send() ──
    # scene_parser fans out to voice_synth + video_gen in parallel
    graph.add_conditional_edges(
        "scene_parser_node",
        parallel_av_router,
        ["voice_synth_node", "video_gen_node"],
    )

    # ── Both parallel branches converge at face_swap ──
    graph.add_edge("voice_synth_node", "face_swap_node")
    graph.add_edge("video_gen_node", "face_swap_node")

    # ── Sequential post-processing ──
    graph.add_edge("face_swap_node", "lip_sync_node")
    graph.add_edge("lip_sync_node", "phase2_output_node")
    graph.add_edge("phase2_output_node", END)

    return graph.compile()