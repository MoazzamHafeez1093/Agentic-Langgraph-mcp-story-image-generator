"""
PROJECT MONTAGE – Phase 1: The Writer's Room
Shared State Schema for LangGraph
"""
from typing import Optional, List, Dict, Any
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Shared state passed between all LangGraph nodes.
    Every field is optional so nodes only update what they own.
    """
    # ── Input ──────────────────────────────────────────────────
    mode: str                          # "manual" | "autonomous"
    raw_input: str                     # User's raw text (script or prompt)

    # ── Script Pipeline ────────────────────────────────────────
    validation_errors: List[str]       # Issues found by validator
    validated_script: Dict[str, Any]   # Structured script after validation
    scene_manifest: List[Dict]         # Final list of scene objects

    # ── Character Pipeline ─────────────────────────────────────
    characters: List[str]              # Character names extracted from script
    character_db: Dict[str, Any]       # Full character identity store

    # ── Image Pipeline ─────────────────────────────────────────
    image_paths: List[str]             # Paths to generated character images

    # ── Control Flow ───────────────────────────────────────────
    hitl_approved: bool                # Human approval flag
    hitl_feedback: str                 # Optional human feedback text
    error: Optional[str]               # Error message if any step fails
    memory_committed: bool             # Whether memory was persisted
    messages: List[str]                # Running log of agent messages
