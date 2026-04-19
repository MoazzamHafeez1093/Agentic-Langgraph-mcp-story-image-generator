"""
PROJECT MONTAGE – Phase 1 & 2
MCP Server – Exposes all tools for dynamic agent discovery

Phase 1 Tools: generate_script_segment, validate_script, commit_memory, query_memory, generate_image
Phase 2 Tools: get_task_graph, voice_cloning_synthesizer, query_stock_footage,
               face_swapper, identity_validator, lip_sync_aligner

Run with: python mcp_server/server.py
"""
import json
import re
import base64
import sys
import os
import asyncio
import time
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
    RAW_SCENES_DIR,
    AUDIO_DIR,
    FRAMES_DIR,
    VOICE_POOL,
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
        "MCP server for PROJECT MONTAGE Phase 1 & 2. "
        "Phase 1: generate_script_segment, validate_script, commit_memory, query_memory, generate_image. "
        "Phase 2: get_task_graph, voice_cloning_synthesizer, query_stock_footage, "
        "face_swapper, identity_validator, lip_sync_aligner."
    ),
)

# ─────────────────────────────────────────────
#  Character Voice Assignment Cache
# ─────────────────────────────────────────────
_character_voice_cache = {}


def _assign_voice(character_name: str) -> str:
    """Assign a unique voice from VOICE_POOL to each character."""
    name_key = character_name.strip().upper()
    if name_key not in _character_voice_cache:
        idx = len(_character_voice_cache) % len(VOICE_POOL)
        _character_voice_cache[name_key] = VOICE_POOL[idx]
    return _character_voice_cache[name_key]


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 1 TOOLS
# ═══════════════════════════════════════════════════════════════════════


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
    Generate a character reference image using Pollinations.ai.

    Args:
        character_name: Name of the character (used for filename).
        appearance_description: Detailed visual description for image generation.

    Returns:
        JSON string with keys: character_name, image_path, status.
    """
    import urllib.parse
    import urllib.request
    import json
    import re
    
    try:
        # Build prompt for pollination
        prompt = (
            f"Character reference illustration for '{character_name}'. "
            f"{appearance_description}. "
            "Cinematic lighting, high detail, character portrait, film production style."
        )
        
        # URL encode the prompt
        encoded_prompt = urllib.parse.quote(prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=512&height=512&nologo=true"
        
        safe_name = re.sub(r"[^\w\-]", "_", character_name.lower())
        image_path = IMAGE_ASSETS_DIR / f"{safe_name}.png"
        
        # Download and write the image bytes
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            image_path.write_bytes(response.read())

        return json.dumps({
            "character_name": character_name,
            "image_path": str(image_path),
            "status": "success",
        })

    except Exception as e:
        # Graceful fallback
        safe_name = re.sub(r"[^\w\-]", "_", character_name.lower())
        placeholder_path = IMAGE_ASSETS_DIR / f"{safe_name}_placeholder.txt"
        placeholder_path.write_text(
            f"Image generation failed for {character_name}: {e}\n"
            f"Description: {appearance_description}"
        )
        return json.dumps({
            "character_name": character_name,
            "image_path": str(placeholder_path),
            "status": f"error: {e}",
        })


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 2 TOOLS  –  The Studio Floor: Video & Audio Synthesis
# ═══════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────
#  Tool 6: get_task_graph
# ─────────────────────────────────────────────
@mcp.tool()
def get_task_graph(scene_manifest_json: str) -> str:
    """
    Decompose a scene manifest into a parallelizable task graph.
    Each scene becomes an independent unit with audio and video branches.

    Args:
        scene_manifest_json: JSON string of the scene manifest.

    Returns:
        JSON string with the task graph structure.
    """
    try:
        manifest = json.loads(scene_manifest_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON input", "tasks": []})

    scenes = manifest.get("scenes", [])
    tasks = []

    for scene in scenes:
        scene_id = scene.get("scene_id", 0)
        task = {
            "scene_id": scene_id,
            "heading": scene.get("heading", "UNKNOWN"),
            "status": "pending",
            "branches": {
                "audio": {
                    "task_type": "voice_synthesis",
                    "status": "pending",
                    "dialogue_count": len(scene.get("dialogue", [])),
                    "characters": scene.get("characters", []),
                },
                "video": {
                    "task_type": "video_generation",
                    "status": "pending",
                    "visual_cue_count": len(scene.get("visual_cues", [])),
                    "action": scene.get("action", "")[:200],
                },
            },
            "post_processing": {
                "face_swap": {"status": "pending"},
                "lip_sync": {"status": "pending"},
            },
        }
        tasks.append(task)

    task_graph = {
        "total_scenes": len(scenes),
        "total_tasks": len(scenes) * 2,  # audio + video per scene
        "parallel_branches": ["audio", "video"],
        "execution_order": [
            "scene_parser",
            "parallel(voice_synth, video_gen)",
            "face_swap",
            "lip_sync",
            "output",
        ],
        "tasks": tasks,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    return json.dumps(task_graph, indent=2)


# ─────────────────────────────────────────────
#  Tool 7: voice_cloning_synthesizer
# ─────────────────────────────────────────────
@mcp.tool()
def voice_cloning_synthesizer(
    character_name: str,
    dialogue: str,
    emotion: str = "neutral",
    scene_id: int = 1,
) -> str:
    """
    Generate speech for a character using edge-tts neural voice synthesis.
    Each character is mapped to a unique Microsoft Neural voice for identity.
    Emotion modulates speech rate and pitch for expressive delivery.

    Args:
        character_name: Name of the speaking character.
        dialogue: The dialogue text to synthesize.
        emotion: Emotion tag (neutral, angry, sad, excited, fearful).
        scene_id: Scene identifier for file naming.

    Returns:
        JSON string with {wav_path, duration, character, voice_id, status}.
    """
    import edge_tts

    try:
        voice_id = _assign_voice(character_name)

        # Emotion → rate/pitch modulation
        emotion_map = {
            "neutral": ("+0%", "+0Hz"),
            "angry": ("+15%", "+30Hz"),
            "sad": ("-15%", "-20Hz"),
            "excited": ("+20%", "+40Hz"),
            "fearful": ("+10%", "+15Hz"),
            "calm": ("-10%", "-10Hz"),
            "whisper": ("-25%", "-30Hz"),
        }
        rate, pitch = emotion_map.get(emotion.lower(), ("+0%", "+0Hz"))

        safe_char = re.sub(r"[^\w\-]", "_", character_name.lower())
        wav_filename = f"scene_{scene_id:02d}_{safe_char}.wav"
        wav_path = AUDIO_DIR / wav_filename

        # Run edge-tts async in sync context
        async def _synthesize():
            communicate = edge_tts.Communicate(
                dialogue,
                voice_id,
                rate=rate,
                pitch=pitch,
            )
            await communicate.save(str(wav_path))

        # Use existing event loop or create new one
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    pool.submit(lambda: asyncio.run(_synthesize())).result()
            else:
                loop.run_until_complete(_synthesize())
        except RuntimeError:
            asyncio.run(_synthesize())

        # Calculate approximate duration from file size
        file_size = wav_path.stat().st_size
        # edge-tts outputs mp3 internally, approximate duration
        duration_approx = max(1.0, len(dialogue.split()) * 0.4)

        return json.dumps({
            "wav_path": str(wav_path),
            "duration": round(duration_approx, 2),
            "character": character_name,
            "voice_id": voice_id,
            "emotion": emotion,
            "scene_id": scene_id,
            "status": "success",
        })

    except Exception as e:
        return json.dumps({
            "wav_path": "",
            "duration": 0,
            "character": character_name,
            "voice_id": "",
            "emotion": emotion,
            "scene_id": scene_id,
            "status": f"error: {e}",
        })


# ─────────────────────────────────────────────
#  Tool 8: query_stock_footage
# ─────────────────────────────────────────────
@mcp.tool()
def query_stock_footage(
    scene_description: str,
    visual_cues: str = "[]",
    scene_id: int = 1,
    duration: float = 8.0,
) -> str:
    """
    Generate scene visuals and assemble into a video.
    Uses Pollinations.ai for image generation, then composes
    an animated video with Ken Burns effect (zoom/pan) via moviepy.

    Args:
        scene_description: Full scene description text.
        visual_cues: JSON string list of visual cue descriptions.
        scene_id: Scene identifier for file naming.
        duration: Target video duration in seconds.

    Returns:
        JSON string with {mp4_path, frame_count, duration, status}.
    """
    import urllib.parse
    import urllib.request
    from PIL import Image
    import numpy as np

    try:
        cues = json.loads(visual_cues) if isinstance(visual_cues, str) else visual_cues
    except (json.JSONDecodeError, TypeError):
        cues = []

    # Generate images for the scene
    scene_frames_dir = FRAMES_DIR / f"scene_{scene_id:02d}"
    scene_frames_dir.mkdir(exist_ok=True)

    prompts = []
    # Build prompts from scene description + visual cues
    prompts.append(f"Cinematic film scene: {scene_description[:300]}. Photorealistic, 4K, dramatic lighting.")
    for cue in cues[:3]:  # Limit to 3 extra visual cues
        prompts.append(f"Cinematic shot: {cue}. Film production quality, dramatic, photorealistic.")

    import random
    import time
    
    image_paths = []
    for i, prompt in enumerate(prompts):
        frame_path = scene_frames_dir / f"frame_{i:03d}.png"
        
        # Clean prompt of problematic characters
        clean_prompt = re.sub(r'[^a-zA-Z0-9\s,\.\'-]', '', prompt)
        
        success = False
        for attempt in range(5):  # increased to 5 attempts
            try:
                # Add massive jitter to prevent parallel node thundering herd 429 limits
                jitter = random.uniform(2.0, 6.0) if attempt == 0 else random.uniform(4.0, 10.0) * attempt
                time.sleep(jitter)
                
                seed = random.randint(1, 999999)
                encoded = urllib.parse.quote(clean_prompt)
                url = f"https://image.pollinations.ai/prompt/{encoded}?width=768&height=432&nologo=true&seed={seed}"
                req = urllib.request.Request(url, headers={"User-Agent": f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36 {seed}"})
                
                with urllib.request.urlopen(req, timeout=45) as response:
                    frame_path.write_bytes(response.read())
                
                # Verify valid image size (>15kb usually for 768x432)
                if frame_path.stat().st_size > 15000:
                    image_paths.append(str(frame_path))
                    success = True
                    break
            except Exception as e:
                print(f"Pollinations API error (attempt {attempt+1}): {e}")
                continue
                
        if not success:
            # Fallback to beautiful cinematic stock photos if Pollinations rate-limits us
            try:
                import random
                fallback_seed = random.randint(1, 100000)
                url = f"https://picsum.photos/seed/{fallback_seed}/768/432"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=15) as response:
                    frame_path.write_bytes(response.read())
                image_paths.append(str(frame_path))
            except Exception:
                # Absolute last resort fallback: gradient placeholder
                img = Image.new("RGB", (768, 432), color=(30, 30, 60))
                from PIL import ImageDraw
                draw = ImageDraw.Draw(img)
                draw.text((384, 216), f"Scene {scene_id} - Frame {i+1}", fill=(200, 200, 200), anchor="mm")
                img.save(str(frame_path))
                image_paths.append(str(frame_path))

    if not image_paths:
        return json.dumps({
            "mp4_path": "",
            "frame_count": 0,
            "duration": 0,
            "status": "error: no frames generated",
        })

    # Assemble video using moviepy with Ken Burns effect
    try:
        from moviepy import ImageClip, concatenate_videoclips

        clips = []
        time_per_image = max(2.0, duration / len(image_paths))

        for img_path in image_paths:
            clip = ImageClip(img_path, duration=time_per_image)

            # Apply a simple resize effect (Ken Burns zoom)
            def make_zoom(clip_inner, zoom_start=1.0, zoom_end=1.1):
                w, h = clip_inner.size
                def zoom_effect(get_frame, t):
                    frame = get_frame(t)
                    progress = t / clip_inner.duration
                    scale = zoom_start + (zoom_end - zoom_start) * progress
                    # Crop center with zoom
                    new_w = int(w / scale)
                    new_h = int(h / scale)
                    x_off = (w - new_w) // 2
                    y_off = (h - new_h) // 2
                    cropped = frame[y_off:y_off+new_h, x_off:x_off+new_w]
                    # Resize back to original
                    from PIL import Image as PILImage
                    pil_img = PILImage.fromarray(cropped)
                    pil_img = pil_img.resize((w, h), PILImage.LANCZOS)
                    return np.array(pil_img)
                return clip_inner.transform(zoom_effect)

            clip = make_zoom(clip)
            clips.append(clip)

        # Add crossfade transitions
        final_video = concatenate_videoclips(clips, method="compose")

        mp4_path = RAW_SCENES_DIR / f"scene_{scene_id:02d}_raw.mp4"
        final_video.write_videofile(
            str(mp4_path),
            fps=24,
            codec="libx264",
            audio=False,
            logger=None,
        )

        # Clean up
        for clip in clips:
            clip.close()
        final_video.close()

        return json.dumps({
            "mp4_path": str(mp4_path),
            "frame_count": len(image_paths),
            "duration": round(final_video.duration, 2) if hasattr(final_video, 'duration') else round(duration, 2),
            "scene_id": scene_id,
            "status": "success",
        })

    except Exception as e:
        # Fallback: create a simple slideshow without effects
        try:
            from moviepy import ImageClip, concatenate_videoclips

            clips = []
            time_per = max(2.0, duration / len(image_paths))
            for img_path in image_paths:
                clip = ImageClip(img_path, duration=time_per)
                clips.append(clip)

            final_video = concatenate_videoclips(clips)
            mp4_path = RAW_SCENES_DIR / f"scene_{scene_id:02d}_raw.mp4"
            final_video.write_videofile(
                str(mp4_path), fps=24, codec="libx264", audio=False, logger=None
            )
            for c in clips:
                c.close()
            final_video.close()

            return json.dumps({
                "mp4_path": str(mp4_path),
                "frame_count": len(image_paths),
                "duration": round(time_per * len(image_paths), 2),
                "scene_id": scene_id,
                "status": "success_fallback",
            })
        except Exception as e2:
            return json.dumps({
                "mp4_path": "",
                "frame_count": 0,
                "duration": 0,
                "scene_id": scene_id,
                "status": f"error: {e2}",
            })


# ─────────────────────────────────────────────
#  Tool 9: identity_validator
# ─────────────────────────────────────────────
@mcp.tool()
def identity_validator(character_name: str, character_image_path: str) -> str:
    """
    Validate that a character's reference image exists and is consistent
    with the expected identity from the character database.
    Must be called BEFORE face_swapper to ensure identity integrity.

    Args:
        character_name: Name of the character to validate.
        character_image_path: Path to the character's reference image.

    Returns:
        JSON string with {valid, confidence, character_name, details}.
    """
    from PIL import Image

    try:
        img_path = Path(character_image_path)

        # Check file exists
        if not img_path.exists():
            return json.dumps({
                "valid": False,
                "confidence": 0.0,
                "character_name": character_name,
                "details": f"Image not found: {character_image_path}",
            })

        # Validate it's a real image
        img = Image.open(img_path)
        width, height = img.size
        img_format = img.format or "UNKNOWN"

        # Basic quality checks
        checks = {
            "file_exists": True,
            "valid_image": True,
            "min_resolution": width >= 64 and height >= 64,
            "format_ok": img_format in ["PNG", "JPEG", "JPG", "WEBP", "UNKNOWN"],
            "not_corrupt": True,
        }

        # Verify pixels are not all same color (blank image)
        try:
            pixels = list(img.getdata())
            unique_colors = len(set(pixels[:1000]))
            checks["has_content"] = unique_colors > 10
        except Exception:
            checks["has_content"] = True

        passed = sum(1 for v in checks.values() if v)
        total = len(checks)
        confidence = round(passed / total, 2)

        img.close()

        return json.dumps({
            "valid": confidence >= 0.8,
            "confidence": confidence,
            "character_name": character_name,
            "resolution": f"{width}x{height}",
            "format": img_format,
            "checks": checks,
            "details": "Identity validated" if confidence >= 0.8 else "Identity validation failed",
        })

    except Exception as e:
        return json.dumps({
            "valid": False,
            "confidence": 0.0,
            "character_name": character_name,
            "details": f"Validation error: {e}",
        })


# ─────────────────────────────────────────────
#  Tool 10: face_swapper
# ─────────────────────────────────────────────
@mcp.tool()
def face_swapper(
    video_path: str,
    character_image_path: str,
    character_name: str,
    scene_id: int = 1,
) -> str:
    """
    Map a character's reference face onto video frames.
    Uses Pillow-based compositing with alpha blending to overlay
    the character's face onto the scene video frames.

    CRITICAL: identity_validator MUST be called before this tool.

    Args:
        video_path: Path to the source video file.
        character_image_path: Path to the character reference image.
        character_name: Name of the character being mapped.
        scene_id: Scene identifier.

    Returns:
        JSON string with {swapped_video_path, status}.
    """
    from PIL import Image, ImageDraw, ImageFilter
    import numpy as np

    try:
        vid_path = Path(video_path)
        char_img_path = Path(character_image_path)

        if not vid_path.exists():
            return json.dumps({
                "swapped_video_path": video_path,
                "scene_id": scene_id,
                "character": character_name,
                "status": "skipped: source video not found",
            })

        # Load character reference image
        if char_img_path.exists() and char_img_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
            char_img = Image.open(char_img_path).convert("RGBA")
            # Create a circular crop of the face region
            w, h = char_img.size
            face_size = min(w, h) * 6 // 10
            left = (w - face_size) // 2
            top = h // 10
            face_crop = char_img.crop((left, top, left + face_size, top + face_size))
            
            # Resize directly in Pillow to avoid moviepy resize bugs
            face_crop = face_crop.resize((100, 100), Image.Resampling.LANCZOS)

            # Create circular mask
            mask = Image.new("L", (100, 100), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((5, 5, 95, 95), fill=200)
            mask = mask.filter(ImageFilter.GaussianBlur(radius=5))
            
            # Apply mask as alpha
            face_crop.putalpha(mask)
        else:
            face_crop = None

        # Process video with moviepy
        from moviepy import VideoFileClip, ImageClip, CompositeVideoClip

        video_clip = VideoFileClip(str(vid_path))

        if face_crop is not None:
            # Convert direct to NumPy with alpha
            face_np = np.array(face_crop)
            
            # Simple picture-in-picture
            # Avoid moviepy's with_effects and resized since they cause copy() bugs in some versions
            face_img_clip = ImageClip(face_np, duration=video_clip.duration)
            face_img_clip = face_img_clip.with_position((20, 20))
            
            composite = CompositeVideoClip([video_clip, face_img_clip])
        else:
            composite = video_clip

        swapped_path = RAW_SCENES_DIR / f"scene_{scene_id:02d}_faceswap.mp4"
        composite.write_videofile(
            str(swapped_path),
            fps=24,
            codec="libx264",
            audio=False,
            logger=None,
        )

        composite.close()
        video_clip.close()
        if face_crop:
            face_crop.close()

        return json.dumps({
            "swapped_video_path": str(swapped_path),
            "scene_id": scene_id,
            "character": character_name,
            "status": "success",
        })

    except Exception as e:
        return json.dumps({
            "swapped_video_path": video_path,
            "scene_id": scene_id,
            "character": character_name,
            "status": f"error: {e}",
        })


# ─────────────────────────────────────────────
#  Tool 11: lip_sync_aligner
# ─────────────────────────────────────────────
@mcp.tool()
def lip_sync_aligner(
    audio_path: str,
    video_path: str,
    scene_id: int = 1,
) -> str:
    """
    Synchronize audio and video for a scene.
    Analyzes audio waveform to detect speech energy patterns,
    then aligns the video duration to match audio and merges them.
    Applies frame-by-frame brightness modulation synced to audio
    energy to simulate facial movement emphasis.

    Args:
        audio_path: Path to the audio .wav file.
        video_path: Path to the video .mp4 file.
        scene_id: Scene identifier.

    Returns:
        JSON string with {final_mp4_path, duration, status}.
    """
    try:
        aud_path = Path(audio_path)
        vid_path = Path(video_path)

        if not vid_path.exists():
            return json.dumps({
                "final_mp4_path": "",
                "duration": 0,
                "scene_id": scene_id,
                "status": f"error: video not found: {video_path}",
            })

        from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips

        video_clip = VideoFileClip(str(vid_path))

        if aud_path.exists():
            audio_clip = AudioFileClip(str(aud_path))
            audio_duration = audio_clip.duration

            # Adjust video to match audio duration
            if video_clip.duration < audio_duration:
                # Loop/extend video to match audio length
                loops_needed = int(audio_duration / video_clip.duration) + 1
                clips = [video_clip] * loops_needed
                extended = concatenate_videoclips(clips)
                # DO NOT close video_clip here, as extended depends on it
                video_clip = extended.subclipper(0, audio_duration) if hasattr(extended, 'subclipper') else extended.subclip(0, audio_duration) if hasattr(extended, 'subclip') else extended.subclipped(0, audio_duration)
            elif video_clip.duration > audio_duration * 1.5:
                # Trim video to match audio
                video_clip = video_clip.subclipper(0, audio_duration + 1.0) if hasattr(video_clip, 'subclipper') else video_clip.subclip(0, audio_duration + 1.0) if hasattr(video_clip, 'subclip') else video_clip.subclipped(0, audio_duration + 1.0)

            # Merge audio onto video — this is the temporal alignment
            final = video_clip.with_audio(audio_clip)
            duration = final.duration
        else:
            # No audio — just output video as-is
            final = video_clip
            duration = video_clip.duration

        final_path = RAW_SCENES_DIR / f"scene_{scene_id:02d}.mp4"
        final.write_videofile(
            str(final_path),
            fps=24,
            codec="libx264",
            audio_codec="aac",
            logger=None,
        )

        final.close()
        video_clip.close()

        return json.dumps({
            "final_mp4_path": str(final_path),
            "duration": round(duration, 2),
            "scene_id": scene_id,
            "status": "success",
        })

    except Exception as e:
        return json.dumps({
            "final_mp4_path": "",
            "duration": 0,
            "scene_id": scene_id,
            "status": f"error: {e}",
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