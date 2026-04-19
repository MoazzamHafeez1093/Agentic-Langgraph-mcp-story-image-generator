"""
PROJECT MONTAGE – Phase 1 & 2
Shared State Schema for LangGraph

Phase 1: The Writer's Room
Phase 2: The Studio Floor – Video and Audio Synthesis Layer
"""
import operator
from typing import Optional, List, Dict, Any, Annotated
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Shared state passed between all LangGraph nodes.
    Every field is optional so nodes only update what they own.

    Phase 2 fields use Annotated[list, operator.add] so that
    parallel Send() branches can safely append to the same keys
    without overwriting each other.
    """
    # ── Phase 1: Input ────────────────────────────────────────
    mode: str                          # "manual" | "autonomous"
    raw_input: str                     # User's raw text (script or prompt)

    # ── Phase 1: Script Pipeline ──────────────────────────────
    validation_errors: List[str]       # Issues found by validator
    validated_script: Dict[str, Any]   # Structured script after validation
    scene_manifest: List[Dict]         # Final list of scene objects

    # ── Phase 1: Character Pipeline ───────────────────────────
    characters: List[str]              # Character names extracted from script
    character_db: Dict[str, Any]       # Full character identity store

    # ── Phase 1: Image Pipeline ───────────────────────────────
    image_paths: List[str]             # Paths to generated character images

    # ── Phase 1: Control Flow ─────────────────────────────────
    hitl_approved: bool                # Human approval flag
    hitl_feedback: str                 # Optional human feedback text
    error: Optional[str]               # Error message if any step fails
    memory_committed: bool             # Whether memory was persisted
    messages: Annotated[List[str], operator.add] # Running log of agent messages

    # ── Phase 2: Task Graph ───────────────────────────────────
    task_graph: List[Dict]             # Decomposed task graph from scene parser

    # ── Phase 2: Audio Pipeline (parallel-safe) ───────────────
    # Each entry: {scene_id, wav_path, duration, character_voice_map}
    audio_outputs: Annotated[List[Dict], operator.add]

    # ── Phase 2: Video Pipeline (parallel-safe) ───────────────
    # Each entry: {scene_id, mp4_path, frame_count, duration}
    video_outputs: Annotated[List[Dict], operator.add]

    # ── Phase 2: Face Swap Pipeline ───────────────────────────
    face_swap_outputs: List[Dict]      # {scene_id, swapped_mp4_path}

    # ── Phase 2: Lip Sync Pipeline ────────────────────────────
    lip_sync_outputs: List[Dict]       # {scene_id, final_mp4_path}

    # ── Phase 2: Fault Tolerance ──────────────────────────────
    phase2_checkpoint: Dict[str, Any]  # Resumability state
