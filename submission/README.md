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

## 🛠️ How It Was Built (The Tech Stack)

| Technology | Implementation How-To & Context |
|------------|---------------------------------|
| **LangGraph** | We relied on `StateGraph` to define rigid nodes (`mode_selector`, `scriptwriter`, `image_synthesizer`). The entire payload routing is tethered via a strict `AgentState` schema using Python's `TypedDict`, ensuring data never hallucinates away between tool calls. |
| **Model Context Protocol (MCP)** | To comply with modern agentic decoupling, NO API calls are made inside the LangGraph nodes. Instead, our agents package requests into JSON-RPC standards and pipe them via `stdio` to an independent background `FastMCP` architecture representing our tools. |
| **Google Gemini 2.0 Flash** | Powers the core narrative creation and structure distillation. We enforce strict JSON coercion within our prompt architectures to guarantee our `.json` deliverables are syntactically bulletproof. |
| **ChromaDB** | Vector persistence storing output mappings. We chose the lightweight `DefaultEmbeddingFunction()` to remove heavy localized PyTorch/HuggingFace dependencies while retaining instantaneous semantic retrieval capabilities. |
| **Stable Diffusion (Pollinations)** | Replaced our initial Imagen fallback architecture dynamically synthesizing visual portraits based strictly on the psychological/visual profile JSON nodes handed to it. |

---

## 🚧 Challenges Faced & Engineering Solutions

1. **The MCP `stdio` Stream Pollution Problem:**
   * **Challenge:** Using `stdio` transport for MCP servers implies that standard output acts as the dedicated API JSON-RPC bridge. We discovered that certain python modules (like the `genai` deprecation warning and FastMCP's ASCII startup banner) were leaking into `sys.stdout` and `sys.stderr`, corrupting the JSON parsing engine and crashing our scriptwriter agent.
   * **Solution:** We aggressively masked `warnings.filterwarnings("ignore")` and implemented a hardened, reversed-line recursive parser to explicitly hunt for the exact `{"jsonrpc": "2.0", "id": 1}` payload inside the corrupted stream buffer, ensuring 100% resilient tool discovery.

2. **The "Paid-Tier API" Asset Pipeline Wall:**
   * **Challenge:** Phase 1 requires character images. While text LLMs are readily accessible on generous free tiers (Gemini), Image Generation models natively demand paid tiers (such as Google’s Imagen-3). The agent would logically crash upon API refusal.
   * **Solution:** We first engineered a graceful "placeholder fallback" tracking `Exceptions` into `.txt` files. However, we stepped it up by isolating a secondary `urllib` HTTP request mapping straight to an un-gated open-source Stable Diffusion proxy (`image.pollinations.ai`). The Node seamlessly drops-in high-quality `.png` assets entirely for free.

3. **Multi-Agent State Hallucinations during Routing:**
   * **Challenge:** Extracting character identities dynamically from unstructured script outputs was confusing the LLM into providing different dialogue keys or dropping the validation loop.
   * **Solution:** Added a discrete `Validator` node using regex and structural parsing, returning boolean safety flags. If a script lacks headings, validation fails natively before saving malicious state, enforcing absolute payload integrity.

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
