# PROJECT MONTAGE – Phase 1: The Writer's Room

> **Course:** Agentic AI CS-4015 | **Assignment 3 Phase 1** | **Due: April 9, 2026**

---

## 🎬 Overview

PROJECT MONTAGE Phase 1 is a **multi-agent autonomous story and image generation system** built with:

| Technology | Purpose |
|---|---|
| **LangGraph** | Stateful agent workflow orchestration |
| **FastMCP** | Dynamic tool discovery (stdio transport) |
| **Google Gemini 2.0 Flash** | LLM for screen writing & character design |
| **Google Gemini Imagen** | Character image generation |
| **ChromaDB** | Persistent vector memory |
| **sentence-transformers** | Text embeddings |

---

## 📁 Project Structure

```
AgenticAI_Ass.03/
├── main.py                    ← Entry point
├── config.py                  ← API keys & paths
├── requirements.txt
├── .env.example               ← Copy to .env and add your API key
├── sample_script.txt          ← Demo screenplay for manual mode
│
├── state/
│   └── schema.py              ← AgentState TypedDict (shared state)
│
├── graph/
│   └── workflow.py            ← LangGraph StateGraph with all 7 nodes
│
├── mcp_server/
│   └── server.py              ← FastMCP server (5 tools)
│
└── outputs/                   ← Generated files
    ├── scene_manifest.json    ← Structured screenplay
    ├── character_db.json      ← Character identity store
    └── image_assets/          ← Character reference images
```

---

## ⚙️ Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Your API Key
```bash
copy .env.example .env
# Edit .env and set GOOGLE_API_KEY=your_key_here
```

---

## 🚀 Running the System

### Interactive Mode (Recommended)
```bash
python main.py
```
Choose between:
- **Option 1**: Autonomous mode – enter a creative story prompt
- **Option 2**: Manual mode – paste a screenplay for validation
- **Option 3**: Demo mode – built-in sci-fi detective story

### CLI Mode
```bash
# Autonomous (prompt-driven)
python main.py --mode autonomous --prompt "A space opera about a rebel AI colony"

# Manual (screenplay validation)
python main.py --mode manual --script sample_script.txt

# Demo
python main.py --demo
```

---

## 🤖 Agent Architecture

```
User Input
    │
    ▼
[Mode Selector Node]
    │
    ├── Manual Script ──► [Validator Node] ──► (invalid?) ──► [Scriptwriter Node]
    │                          │ (valid)                              │
    │                          ▼                                      │
    └── Prompt ──────────────────────────────────────────────────────►
                               │
                               ▼
                          [HITL Node] ─── reject ──► END
                               │ approve
                               ▼
                       [Character Node]
                               │
                               ▼
                        [Image Node]
                               │
                               ▼
                    [Memory Commit Node]
                               │
                               ▼
                             END
```

---

## 🔧 MCP Tools (Dynamically Discovered)

All tools are registered on the FastMCP server and discovered at runtime:

| Tool | Description |
|---|---|
| `generate_script_segment` | Generate multi-scene screenplay from prompt |
| `validate_script` | Check scene headers, dialogue labels, actions |
| `commit_memory` | Store embeddings in ChromaDB |
| `query_memory` | Semantic retrieval from ChromaDB |
| `generate_image` | Generate character images via Gemini Imagen |

---

## 📤 Outputs

After a successful run:

| File | Contents |
|---|---|
| `outputs/scene_manifest.json` | Full structured screenplay with scenes, dialogue, visual cues |
| `outputs/character_db.json` | Character profiles: personality, appearance, backstory |
| `outputs/image_assets/*.png` | Generated character reference images |

---

## 📊 Evaluation Mapping

| Criteria | Implementation | Marks |
|---|---|---|
| Agent Definition | 6 agents with clear roles & reasoning loops in `graph/workflow.py` | 20 |
| Script Generation Quality | Gemini-generated multi-scene JSON with dialogue & visual cues | 15 |
| MCP Integration | All 5 tools via FastMCP stdio, no hardcoded API calls | 15 |
| LangGraph Workflow | 7-node StateGraph with conditional routing | 10 |
| Human-in-the-Loop | `hitl_node` pauses & awaits approve/reject | 10 |
| Output Completeness | JSON files + images generated every run | 5 |
| **Total** | | **75** |
