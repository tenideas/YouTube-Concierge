# Insight — AI-Powered YouTube Knowledge Engine

**Insight** turns YouTube into an interactive knowledge source. Instead of scrubbing through 30–90 minute videos, you can ask questions, request comparisons across videos, summarize content intelligently, and continue follow‑up conversations without restating context.

This project is built on Google’s **Agent Development Kit (ADK)** and powered by **Gemini 2.5**, using a Planner‑Executor agent architecture optimized for long‑form media understanding.

---

## 🚀 What Insight Does

- Treats YouTube like a **queryable database**
- Summarizes long videos with structure and topic awareness
- Answers questions using transcript‑grounded retrieval (RAG)
- Remembers active video context for natural follow‑ups
- Compares videos side‑by‑side (argument vs argument, topic vs topic)
- Handles genre‑specific reading styles (vlog vs lecture vs music)

If your goal is **speed, depth, or interactive exploration of content**, Insight removes friction.

---

## 🧠 Architecture Overview

Insight uses a modular multi‑agent design coordinated by a central Planner.

```mermaid
graph TD
    User[User Input] --> Agent[Planner]
    Agent -->|Produces Task Plan| Tools

    subgraph Cognitive Layer
        Classifier[Classifier]
        Summarizer[Summarizer]
        QA[Question Answering]
        History[History Manager]
    end

    subgraph Infrastructure Layer
        Memory[Memory Service (ADK)]
        YouTube[YouTube Transcript API]
        Cache[Local JSON Cache]
    end

    Tools --> Classifier
    Tools --> Summarizer
    Tools --> QA
    Tools --> Memory

    Classifier --> Gemini
    Summarizer --> Gemini
    QA --> Gemini

    Memory --> LocalStore[(Local State)]
```

💡 **How it works (practical flow example)**  
1. You ask a question about a video  
2. The Planner generates an execution plan  
3. Transcript is fetched, classified, summarized or queried  
4. Insight returns structured information instead of timestamps and guesses  
5. You follow up naturally — session context persists

---

## ✨ Feature Highlights

| Feature | Why It Matters |
|--------|----------------|
| Planner‑Executor pipeline | Breaks user intent into actionable tool calls |
| Transcript‑aware RAG QA | Answers grounded in the actual video content |
| Genre‑adaptive prompting | Scientific talk ≠ vlog ≠ gaming commentary |
| Memory + Sessions | Ask follow‑ups without providing URLs repeatedly |
| History compaction | Unlimited session length without token bloat |
| Logging & debug visibility | Every decision recorded in `agent_####.log` |

---

## 📦 Installation

### Prerequisites
- Python **3.9+**
- Google Cloud project + Gemini API access
- API key from Google AI Studio

### Setup

```bash
git clone https://github.com/yourusername/insight-agent.git
cd insight-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Add credentials:

```bash
echo "GOOGLE_API_KEY=YOUR_KEY_HERE" > .env
```

### Verify installation

```bash
python -m cli.main --help
```

If this prints the command interface — you're set.

---

## 🔥 Usage Examples

### Summarize a video
```bash
python -m cli.main

# Once the interactive prompt starts:
Request> Summarize https://youtube.com/watch?v=dQw4w9WgXcQ
...
Request> Is Rick planning to give up?
```

### Multi‑turn memory
```
Request> Summarize this lecture
Request> Now extract all key formulas
Request> Which one relates to entropy?
```

Insight remembers — no URL repetition required.

---

## 📂 Project Structure

```
app/                Core agent + planner logic
services/           Classifier / Summarizer / RAG / Memory
infra/              Gemini client + YouTube interface
config/             Prompt templates & runtime settings
cli/                Command line entry point
logs/               Step‑by‑step agent reasoning traces
```

### For contributors
- Start with `app/agent.py`
- Extend functionality by adding a tool or service
- Prompts live under `config/prompts.py`
- PRs should include test coverage where meaningful

---

## ⚠️ Known Considerations

- Transcript quality varies with YouTube availability  
- Long videos may require chunked fetch + compaction  
- Google API rate limits may apply  

A troubleshooting section is available in the wiki.

---

## 📄 License
MIT — free to use, modify, and build on.

---