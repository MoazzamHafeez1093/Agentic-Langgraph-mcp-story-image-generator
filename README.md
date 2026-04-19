<div align="center">
  
# 🎬 PROJECT MONTAGE
### Phase 1: The Writer's Room | Phase 2: The Studio Floor

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Agents-orange?style=for-the-badge)](https://python.langchain.com/docs/langgraph)
[![MCP](https://img.shields.io/badge/MCP-Protocol-purple?style=for-the-badge)](https://github.com/microsoft/multi-agent-frameworks)
[![Gemini](https://img.shields.io/badge/Google-Gemini_2.5_Flash-success?style=for-the-badge&logo=google)](https://deepmind.google/technologies/gemini/)

A cutting-edge **multi-agent orchestration framework** using **LangGraph** and the **Model Context Protocol (MCP)**. Phase 1 simulates a Hollywood "Writer's Room" — autonomously generating screenplays and character art. Phase 2 implements "The Studio Floor" — a **parallel multi-agent system** that transforms structured narrative into **synchronized audiovisual content**.

</div>

---

## 🌟 Key Features

### Phase 1: The Writer's Room
* 🧠 **Multi-Agent Orchestration**: Stateful graph delegation between 5 isolated agents (Selector, Validator, Scriptwriter, Designer, Synthesizer).
* 🔌 **Dynamic MCP Discovery**: All LLM cognitive abilities are delegated into an isolated FastMCP server using `stdio` transport.
* ⏸️ **Human-in-the-Loop (HITL)**: Built-in checkpoints pausing the graph before character generation for director approvals.
* 🎨 **Autonomous Asset Synthesis**: Generates character reference images via Pollinations.ai (free Stable Diffusion proxy).
* 🗄️ **Memory Persistence**: Embedded local **ChromaDB** tracks all synthesized characters and narrative sequences.

### Phase 2: The Studio Floor
* 🎤 **Voice Synthesis**: Emotion-aware TTS using Microsoft Neural voices (edge-tts) with per-character voice identity.
* 🎬 **Video Generation**: Scene visuals generated via Pollinations.ai, assembled into animated videos with Ken Burns effects.
* 🎭 **Face Mapping**: Character reference images composited onto video frames with identity validation.
* 👄 **Lip Sync**: Audio-video temporal alignment with frame-by-frame synchronization.
* ⚡ **Parallel Processing**: Audio and video branches execute **concurrently** via LangGraph's `Send()` API.
* 🛡️ **Fault Tolerance**: Stateful resumability with `commit_memory` checkpoints at every stage.

---

## 🛠️ Tech Stack

| Technology | Role |
|------------|------|
| **LangGraph** | `StateGraph` with `Send()` API for parallel branching |
| **Model Context Protocol (MCP)** | 11 tools exposed via FastMCP (5 Phase 1 + 6 Phase 2) |
| **Google Gemini 2.5 Flash** | Script generation, character profiling |
| **ChromaDB** | Vector persistence for memory and fault tolerance |
| **Pollinations.ai** | Free image & scene generation (no API key required) |
| **edge-tts** | Microsoft Neural TTS with emotion-aware speech synthesis |
| **moviepy** | Video composition, Ken Burns effects, A/V merging |
| **Pillow** | Face compositing and identity validation |

---

## 🏗️ Architecture

### Phase 2 Parallel Processing Pipeline

```
scene_manifest.json
        │
  ┌─────▼─────┐
  │Scene Parser│  ← get_task_graph, commit_memory
  └─────┬─────┘
        │
   Send() API          ← PARALLEL BRANCHING
   ┌────┴────┐
   │         │
┌──▼──┐  ┌──▼───┐
│Voice│  │Video │     ← voice_cloning_synthesizer
│Synth│  │ Gen  │     ← query_stock_footage
└──┬──┘  └──┬───┘
   │         │
   └────┬────┘         ← CONVERGENCE
        │
  ┌─────▼─────┐
  │ Face Swap │       ← identity_validator + face_swapper
  └─────┬─────┘
        │
  ┌─────▼─────┐
  │ Lip Sync  │       ← lip_sync_aligner
  └─────┬─────┘
        │
  ┌─────▼─────┐
  │  Output   │       ← raw_scenes/*.mp4
  └───────────┘
```

---

## 📂 Project Structure

```text
├── agents/             # Node definitions for LangGraph
├── graph/
│   └── workflow.py     # Core StateGraph: Phase 1 + Phase 2 workflows
├── mcp_server/
│   └── server.py       # FastMCP server (11 tools: Phase 1 + Phase 2)
├── state/
│   └── schema.py       # TypedDict AgentState with Annotated parallel fields
├── outputs/
│   ├── image_assets/       # .png character artwork
│   ├── raw_scenes/         # .mp4 final scene videos (Phase 2)
│   ├── audio/              # .wav voice tracks (Phase 2)
│   ├── frames/             # Intermediate frame sequences (Phase 2)
│   ├── scene_manifest.json # Compiled film skeleton
│   ├── character_db.json   # JSON identity mappings
│   └── task_graph_log.json # Task decomposition log (Phase 2)
├── config.py           # Core variables, paths, voice mappings
├── main.py             # CLI Launch Interface
├── requirements.txt    # Python dependencies
└── README.md           # You are here!
```

---

## 🚀 Getting Started

### 1. Requirements & Setup
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY="AIzaSyYourSecretKeyHere..."
```

### 3. Execution

#### Phase 1 Only (Script + Character Generation)
```bash
python main.py --demo
```

#### Phase 2 Only (Video + Audio Synthesis)
```bash
python main.py --phase2
```

#### Full End-to-End Pipeline
```bash
python main.py --full --demo
```

#### Resume from Crash
```bash
python main.py --phase2 --resume
```

#### Interactive Mode
```bash
python main.py
```
Presents options:
1. **Autonomous** – AI generates screenplay from your prompt
2. **Manual** – Paste your own screenplay
3. **Demo** – Built-in psychological thriller demo
4. **Phase 2** – Run Studio Floor on existing manifest
5. **Full Demo** – Phase 1 → Phase 2 end-to-end

---

## 🤖 MCP Tools Reference

### Phase 1 Tools
| Tool | Description |
|------|-------------|
| `generate_script_segment` | Generates structured multi-scene screenplay |
| `validate_script` | Validates manually provided screenplays |
| `commit_memory` | Stores data in ChromaDB vector store |
| `query_memory` | Retrieves semantically similar documents |
| `generate_image` | Generates character reference images |

### Phase 2 Tools
| Tool | Description |
|------|-------------|
| `get_task_graph` | Decomposes scenes into parallelizable task graph |
| `voice_cloning_synthesizer` | Emotion-aware TTS with per-character voices |
| `query_stock_footage` | Generates scene visuals → animated video |
| `identity_validator` | Validates character identity before face mapping |
| `face_swapper` | Composites character faces onto video frames |
| `lip_sync_aligner` | Synchronizes audio waveform with video frames |

---

## 📜 Output Deliverables

### Phase 1
* `scene_manifest.json` — Structured scene representations
* `character_db.json` — Character identity mappings
* `image_assets/*.png` — Character reference images

### Phase 2
* `raw_scenes/scene_XX.mp4` — Final lip-synced scene videos
* `audio/scene_XX_*.wav` — Voice synthesis audio tracks
* `task_graph_log.json` — Task decomposition log
* `phase2_checkpoint.json` — Resumability checkpoint

---

## 📊 Evaluation Coverage

| Criteria | Marks | Implementation |
|----------|-------|---------------|
| Parallel Architecture | 10 | `Send()` API fan-out for audio + video branches |
| Audio Quality | 20 | edge-tts Neural voices with emotion modulation |
| Video Quality | 20 | Pollinations.ai images + moviepy Ken Burns animation |
| Lip Sync Accuracy | 10 | Audio-video temporal alignment via moviepy |
| MCP Tool Usage | 5 | 11 tools via FastMCP with dynamic discovery |
| Fault Tolerance | 5 | `commit_memory` checkpoints + `--resume` flag |
| **Total** | **70** | |

---

<div align="center">
<i>Crafted for the Advanced Agentic Coding Architecture Challenge</i>
</div>
