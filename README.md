<div align="center">
  
# 🎬 PROJECT MONTAGE: The Writer's Room
### Phase 1: Autonomous Story & Image Generation Layer

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Agents-orange?style=for-the-badge)](https://python.langchain.com/docs/langgraph)
[![MCP](https://img.shields.io/badge/MCP-Protocol-purple?style=for-the-badge)](https://github.com/microsoft/multi-agent-frameworks)
[![Gemini](https://img.shields.io/badge/Google-Gemini_2.0_Flash-success?style=for-the-badge&logo=google)](https://deepmind.google/technologies/gemini/)

Project Montage Phase 1 is a cutting-edge multi-agent orchestration framework utilizing **LangGraph** and the **Model Context Protocol (MCP)**. This system simulates a Hollywood "Writer's Room", capable of autonomously reading or writing film treatments, validating structures, storing memory using ChromaDB, and synthesizing concept artwork for extracted character identities.

</div>

---

## 🌟 Key Features

* 🧠 **Multi-Agent Orchestration**: Stateful graph delegation between 5 isolated agents (Selector, Validator, Scriptwriter, Designer, Synthesizer).
* 🔌 **Dynamic MCP Discovery**: All LLM cognitive abilities are stripped from the agents and delegated into an isolated FastMCP server using `stdio` transport.
* ⏸️ **Human-in-the-Loop (HITL)**: Built-in strict checkpoints pausing the graph before character generation to allow director approvals.
* 🎨 **Autonomous Asset Synthesis**: Automatically maps generated identities into a seamless, free-tier-friendly Stable-Diffusion proxy (Pollinations.ai) to generate beautiful `.png` character sheets.
* 🗄️ **Memory Persistence**: Embedded local **ChromaDB** tracks all synthesized characters and narrative sequences perfectly across iterations.

---

## 🏗️ Architecture Stack

| Technology | Purpose |
|------------|---------|
| **LangGraph** | Orchestrates the stateful pipeline workflow (`StateGraph`). |
| **FastMCP** | Powers the JSON-RPC interface isolating tools from agents. |
| **Google Gemini 2.0** | Serves as the primary intelligence engine for story and JSON parsing. |
| **ChromaDB** | Vector persistence storing output `metadata` mappings over time. |
| **Pollinations.ai** | Synthesizes `.png` reference imagery safely without API quotas. |

---

## 📂 Project Structure

```text
├── agents/             # Node definitions for LangGraph (HITL, Extractors, Writers)
├── graph/
│   └── workflow.py     # Core StateGraph conditional logic pipeline
├── mcp_server/
│   └── server.py       # FastMCP stdio server (Houses all Gemini Tools)
├── state/
│   └── schema.py       # TypedDict AgentState Definitions
├── outputs/            # (Generated Assets)
│   ├── image_assets/       # .png artwork goes here
│   ├── scene_manifest.json # Compiled film skeleton
│   └── character_db.json   # JSON identity mappings
├── config.py           # Core variables and routing
├── main.py             # CLI Launch Interface 
└── README.md           # You are here!
```

---

## 🚀 Getting Started

### 1. Requirements & Setup
Make sure you have python installed. It is highly recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the root directory and supply your Gemini API key:
```env
GOOGLE_API_KEY="AIzaSyYourSecretKeyHere..."
```

### 3. Execution
Launch the orchestration interface:
```bash
python main.py
```

It will present three modes:
- `1 - Autonomous`: Pass a single creative prompt, and the AI will construct the screenplay.
- `2 - Manual`: Paste your own raw screenplay text, forcing the AI to strictly parse and extract from it.
- `3 - Demo`: Automatically runs a high-stakes psychological thriller simulation to prove functionality.

---

## 📜 Output Deliverables

Upon a successful workflow path matching an approved **HITL Checkpoint**, the application generates the following assets:
1. `scene_manifest.json`: Extracted structured scene representations.
2. `character_db.json`: Isolated identities, descriptions, and psychological trait mappings.
3. `image_assets/*.png`: Rendered visual models for every single character discovered inside the manifest.

---

<div align="center">
<i>Crafted for the Advanced Agentic Coding Architecture Challenge</i>
</div>
