"""
PROJECT MONTAGE – Phase 1 & Phase 2
Main Entry Point

Phase 1: The Writer's Room   — Story & Image Generation
Phase 2: The Studio Floor    — Video & Audio Synthesis

Usage:
  python main.py                          # Interactive mode (Phase 1)
  python main.py --mode autonomous --prompt "A sci-fi thriller about AI"
  python main.py --mode manual --script script.txt
  python main.py --demo                   # Run Phase 1 demo

  python main.py --phase2                 # Run Phase 2 on existing manifest
  python main.py --phase2 --resume        # Resume Phase 2 from checkpoint
  python main.py --full --demo            # Run Phase 1 + Phase 2 end-to-end
"""
import argparse
import sys
import json
import io
from pathlib import Path

# Force stdout to UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from graph.workflow import build_workflow, build_phase2_workflow
from config import (
    SCENE_MANIFEST_PATH,
    CHARACTER_DB_PATH,
    IMAGE_ASSETS_DIR,
    PHASE2_CHECKPOINT_PATH,
)


BANNER_P1 = """
====================================================================
          PROJECT MONTAGE – Phase 1: THE WRITER'S ROOM            
   Autonomous Story & Image Generation Layer  |  LangGraph + MCP  
====================================================================
"""

BANNER_P2 = """
====================================================================
          PROJECT MONTAGE – Phase 2: THE STUDIO FLOOR             
   Video & Audio Synthesis Layer  |  Parallel Multi-Agent System   
====================================================================
"""

DEMO_PROMPT = (
    "A psychological thriller set in 2045 where a detective with memory implants "
    "investigates the murder of a world-renowned AI ethicist. The killer may be "
    "an AI itself trying to protect its own existence."
)


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 1 PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def run_phase1(raw_input: str) -> dict:
    """Run the full Phase 1 LangGraph pipeline and return the final state."""
    print(BANNER_P1)

    workflow = build_workflow()
    initial_state = {
        "raw_input": raw_input,
        "mode": "",
        "validation_errors": [],
        "validated_script": {},
        "scene_manifest": [],
        "characters": [],
        "character_db": {},
        "image_paths": [],
        "hitl_approved": False,
        "hitl_feedback": "",
        "error": None,
        "memory_committed": False,
        "messages": [],
    }

    print("  Starting Phase 1 pipeline...\n")
    final_state = workflow.invoke(initial_state)

    # Print message log
    print("\n  PHASE 1 AGENT LOG:")
    for msg in final_state.get("messages", []):
        print(f"   {msg}")

    return final_state


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 2 PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def run_phase2(phase1_state: dict = None, resume: bool = False) -> dict:
    """
    Run Phase 2: The Studio Floor.
    
    Reads scene_manifest.json and character_db.json from Phase 1 outputs,
    then processes through the parallel audio/video synthesis pipeline.
    
    Args:
        phase1_state: Optional state dict from Phase 1 (for end-to-end runs).
        resume: If True, attempt to resume from a saved checkpoint.
    """
    print(BANNER_P2)

    # ── Build initial state from Phase 1 outputs ──
    if phase1_state:
        # End-to-end mode: carry state forward
        initial_state = dict(phase1_state)
    else:
        # Standalone Phase 2: load from disk
        initial_state = {
            "raw_input": "",
            "mode": "",
            "validation_errors": [],
            "validated_script": {},
            "scene_manifest": [],
            "characters": [],
            "character_db": {},
            "image_paths": [],
            "hitl_approved": True,
            "hitl_feedback": "",
            "error": None,
            "memory_committed": True,
            "messages": [],
            "task_graph": [],
            "audio_outputs": [],
            "video_outputs": [],
            "face_swap_outputs": [],
            "lip_sync_outputs": [],
            "phase2_checkpoint": {},
        }

        # Load scene manifest
        if SCENE_MANIFEST_PATH.exists():
            manifest = json.loads(SCENE_MANIFEST_PATH.read_text(encoding="utf-8"))
            initial_state["scene_manifest"] = manifest.get("scenes", [])
            initial_state["validated_script"] = {
                "title": manifest.get("title", "Untitled"),
                "genre": manifest.get("genre", "Unknown"),
                "scenes": manifest.get("scenes", []),
            }
            initial_state["image_paths"] = manifest.get("image_assets", [])
            print(f"  Loaded manifest: {manifest.get('title', 'Untitled')}")
            print(f"  Scenes: {manifest.get('total_scenes', 0)}")
        else:
            print("  ❌ ERROR: scene_manifest.json not found!")
            print(f"  Expected at: {SCENE_MANIFEST_PATH}")
            print("  Run Phase 1 first: python main.py --demo")
            return {}

        # Load character DB
        if CHARACTER_DB_PATH.exists():
            char_db = json.loads(CHARACTER_DB_PATH.read_text(encoding="utf-8"))
            initial_state["character_db"] = char_db.get("characters", {})
            initial_state["characters"] = list(char_db.get("characters", {}).keys())
            print(f"  Characters: {len(initial_state['characters'])}")
        else:
            print("  ⚠️ WARNING: character_db.json not found, proceeding without characters")

    # ── Check for resume ──
    if resume and PHASE2_CHECKPOINT_PATH.exists():
        checkpoint = json.loads(PHASE2_CHECKPOINT_PATH.read_text(encoding="utf-8"))
        print(f"\n  📋 Found checkpoint from {checkpoint.get('completed_at', 'unknown')}")
        print(f"     Audio: {checkpoint.get('audio_generated', 0)}, "
              f"Video: {checkpoint.get('videos_generated', 0)}, "
              f"Lip-synced: {checkpoint.get('lip_synced', 0)}")
        
        proceed = input("\n  Resume from checkpoint? (y/n): ").strip().lower()
        if proceed == "y":
            initial_state["phase2_checkpoint"] = checkpoint
            print("  Resuming from checkpoint...\n")
        else:
            print("  Starting fresh...\n")

    # Ensure Phase 2 list fields exist
    for key in ["task_graph", "audio_outputs", "video_outputs", 
                 "face_swap_outputs", "lip_sync_outputs"]:
        if key not in initial_state:
            initial_state[key] = []
    if "phase2_checkpoint" not in initial_state:
        initial_state["phase2_checkpoint"] = {}

    # ── Build and run Phase 2 workflow ──
    print("  Starting Phase 2 pipeline...\n")
    workflow = build_phase2_workflow()
    final_state = workflow.invoke(initial_state)

    # Print message log
    print("\n  PHASE 2 AGENT LOG:")
    for msg in final_state.get("messages", []):
        print(f"   {msg}")

    return final_state


# ═══════════════════════════════════════════════════════════════════════
#  INTERACTIVE MODE
# ═══════════════════════════════════════════════════════════════════════

def interactive_mode() -> None:
    """Prompt the user to choose a mode and enter input."""
    print(BANNER_P1)
    print("Select mode:")
    print("  1) Autonomous  – provide a creative prompt")
    print("  2) Manual      – paste a screenplay")
    print("  3) Demo        – run built-in demo prompt")
    print("  4) Phase 2     – run Studio Floor on existing manifest")
    print("  5) Full Demo   – run Phase 1 + Phase 2 end-to-end")
    print()

    choice = input("Enter 1, 2, 3, 4, or 5: ").strip()

    if choice == "1":
        prompt = input("\n  Enter your story prompt:\n> ").strip()
        if not prompt:
            print("No prompt provided. Exiting.")
            sys.exit(1)
        run_phase1(prompt)

    elif choice == "2":
        print("\n  Paste your screenplay below.")
        print("    When done, type END on a new line and press Enter.\n")
        lines = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
        script_text = "\n".join(lines).strip()
        if not script_text:
            print("No script provided. Exiting.")
            sys.exit(1)
        run_phase1(script_text)

    elif choice == "3":
        print(f"\n  Demo prompt: \"{DEMO_PROMPT}\"\n")
        run_phase1(DEMO_PROMPT)

    elif choice == "4":
        run_phase2()

    elif choice == "5":
        print(f"\n  Full Demo: Phase 1 + Phase 2")
        print(f"  Prompt: \"{DEMO_PROMPT}\"\n")
        state = run_phase1(DEMO_PROMPT)
        if state.get("memory_committed"):
            run_phase2(phase1_state=state)
        else:
            print("  Phase 1 did not complete. Skipping Phase 2.")

    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
#  CLI ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PROJECT MONTAGE – Phase 1 & 2 Pipeline"
    )
    parser.add_argument("--mode", choices=["autonomous", "manual"], help="Input mode")
    parser.add_argument("--prompt", type=str, help="Creative story prompt (autonomous mode)")
    parser.add_argument("--script", type=str, help="Path to screenplay .txt file (manual mode)")
    parser.add_argument("--demo", action="store_true", help="Run built-in demo prompt")
    parser.add_argument("--phase2", action="store_true", help="Run Phase 2 on existing manifest")
    parser.add_argument("--resume", action="store_true", help="Resume Phase 2 from checkpoint")
    parser.add_argument("--full", action="store_true", help="Run Phase 1 + Phase 2 end-to-end")

    args = parser.parse_args()

    if args.phase2:
        # Phase 2 only
        run_phase2(resume=args.resume)

    elif args.full and args.demo:
        # Full end-to-end demo
        print(f"\n  Full Demo: Phase 1 + Phase 2")
        state = run_phase1(DEMO_PROMPT)
        if state.get("memory_committed"):
            run_phase2(phase1_state=state)

    elif args.demo:
        # Phase 1 demo
        print(f"\n  Demo prompt: \"{DEMO_PROMPT}\"\n")
        run_phase1(DEMO_PROMPT)

    elif args.mode == "autonomous" and args.prompt:
        run_phase1(args.prompt)

    elif args.mode == "manual" and args.script:
        script_path = Path(args.script)
        if not script_path.exists():
            print(f"❌  Script file not found: {script_path}")
            sys.exit(1)
        run_phase1(script_path.read_text(encoding="utf-8"))

    else:
        # Fall back to interactive
        interactive_mode()


if __name__ == "__main__":
    main()
