"""
PROJECT MONTAGE – Phase 1: The Writer's Room
Main Entry Point

Usage:
  python main.py                          # Interactive mode
  python main.py --mode autonomous --prompt "A sci-fi thriller about AI"
  python main.py --mode manual --script script.txt
  python main.py --demo                   # Run a built-in demo
"""
import argparse
import sys
import json
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from graph.workflow import build_workflow
from config import SCENE_MANIFEST_PATH, CHARACTER_DB_PATH, IMAGE_ASSETS_DIR


BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║          PROJECT MONTAGE – Phase 1: THE WRITER'S ROOM            ║
║   Autonomous Story & Image Generation Layer  |  LangGraph + MCP  ║
╚══════════════════════════════════════════════════════════════════╝
"""

DEMO_PROMPT = (
    "A psychological thriller set in 2045 where a detective with memory implants "
    "investigates the murder of a world-renowned AI ethicist. The killer may be "
    "an AI itself trying to protect its own existence."
)


def run_pipeline(raw_input: str) -> dict:
    """Run the full LangGraph pipeline and return the final state."""
    print(BANNER)

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

    print("  Starting pipeline...\n")
    final_state = workflow.invoke(initial_state)

    # Print message log
    print("\n  AGENT LOG:")
    for msg in final_state.get("messages", []):
        print(f"   {msg}")

    return final_state


def interactive_mode() -> None:
    """Prompt the user to choose a mode and enter input."""
    print(BANNER)
    print("Select mode:")
    print("  1) Autonomous  – provide a creative prompt")
    print("  2) Manual      – paste a screenplay")
    print("  3) Demo        – run built-in demo prompt")
    print()

    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == "1":
        prompt = input("\n  Enter your story prompt:\n> ").strip()
        if not prompt:
            print("No prompt provided. Exiting.")
            sys.exit(1)
        run_pipeline(prompt)

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
        run_pipeline(script_text)

    elif choice == "3":
        print(f"\n  Demo prompt: \"{DEMO_PROMPT}\"\n")
        run_pipeline(DEMO_PROMPT)

    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PROJECT MONTAGE Phase 1 – The Writer's Room"
    )
    parser.add_argument("--mode", choices=["autonomous", "manual"], help="Input mode")
    parser.add_argument("--prompt", type=str, help="Creative story prompt (autonomous mode)")
    parser.add_argument("--script", type=str, help="Path to screenplay .txt file (manual mode)")
    parser.add_argument("--demo", action="store_true", help="Run built-in demo prompt")

    args = parser.parse_args()

    if args.demo:
        print(f"\n  Demo prompt: \"{DEMO_PROMPT}\"\n")
        run_pipeline(DEMO_PROMPT)

    elif args.mode == "autonomous" and args.prompt:
        run_pipeline(args.prompt)

    elif args.mode == "manual" and args.script:
        script_path = Path(args.script)
        if not script_path.exists():
            print(f"❌  Script file not found: {script_path}")
            sys.exit(1)
        run_pipeline(script_path.read_text(encoding="utf-8"))

    else:
        # Fall back to interactive
        interactive_mode()


if __name__ == "__main__":
    main()
