# 🎬 AutoStream AI Agent
### Social-to-Lead Agentic Workflow | Built with LangGraph + Groq (Llama 3)

---

## 🎥 Demo Video

A short walkthrough demonstrating:
- Intent detection  
- RAG-based pricing response  
- High-intent identification  
- Lead collection and tool execution  

👉 [Watch Demo Video](https://drive.google.com/file/d/1zDqkKr76_y1V5b9PvPWAGjmXRNFz3Bl5/view?usp=sharing)

## 📌 Overview

AutoStream AI Agent is a production-ready conversational AI agent for **AutoStream** — a fictional SaaS platform that provides automated video editing tools for content creators.

This agent is **not a simple chatbot**. It is an agentic workflow that:
- Understands user **intent** in real-time
- Answers product and pricing questions using **RAG** (Retrieval-Augmented Generation) from a local knowledge base
- Detects **high-intent leads** and collects their details
- Triggers a **lead capture tool** only after all required information is collected
- Maintains **full memory** across 5–6 conversation turns using LangGraph state

---

## ✨ Features

- **Intent Classification** — Classifies every message into: `greeting`, `product_inquiry`, or `high_intent` using a deterministic regex-based classifier
- **RAG Pipeline** — Retrieves accurate answers from a local JSON knowledge base. Zero hallucination by design — the LLM is strictly instructed to answer only from retrieved context
- **Lead Capture Tool** — Collects name, email, and creator platform incrementally. Tool fires **only** after all three fields are validated
- **Multi-turn Memory** — Full conversation history is carried in `AgentState` and passed to the LLM on every turn
- **LangGraph Workflow** — Clean FSM-style routing with 5 nodes and a conditional edge router
- **Input Validation** — Email regex validation, name length check, platform extraction
- **CLI Interface** — Color-coded terminal output with debug state display

---

## 🏗️ Architecture

### Why LangGraph?

This agent is a **stateful workflow with branching logic** — not a simple chain. Each user turn can transition the agent between distinct stages (`idle → answering → collecting → captured`), and these transitions depend on both the current message *and* all prior context. LangGraph's `StateGraph` models this naturally as a directed graph with typed, persistent state flowing between nodes. It also makes the routing logic explicit and easy to extend.

### How State Works

A single `AgentState` TypedDict is the source of truth for the entire conversation. It persists across every turn and carries:
- `messages` — full conversation history for LLM coherence
- `intent` — the classified intent for the current turn
- `stage` — FSM stage: `idle`, `answering`, `collecting`, `capturing`, `captured`
- `lead` — incrementally populated dict of `name`, `email`, `platform`
- `pending_field` — which field the agent is currently waiting to collect

### How Routing Works

Every turn enters at `classify_intent`. The `router` conditional edge reads both `intent` and `stage` to decide the next node. Stage takes precedence over intent — if `stage == "collecting"`, the agent keeps collecting regardless of what the user says, ensuring no premature tool calls. Once all three lead fields are collected, `stage` transitions to `"capturing"` and the graph routes to `capture_lead`.

### RAG Approach

Keyword-based retrieval over a local `knowledge_base.json` file. Each knowledge entry has a `keywords` list; the retriever scores entries by keyword overlap and injects the top matches as context into the LLM system prompt. The LLM is explicitly forbidden from adding information beyond the retrieved context.

---

## 📁 Project Structure

```
autostream/
│
├── main.py                  # CLI entry point and conversation loop
│
├── agent/
│   ├── __init__.py          # Package exports
│   ├── graph.py             # LangGraph StateGraph + router logic
│   ├── state.py             # AgentState TypedDict (shared memory object)
│   ├── nodes.py             # 5 node functions: greet, RAG, collect, capture
│   ├── intent.py            # Regex-based intent classifier
│   ├── rag.py               # RAG retrieval pipeline
│   └── tools.py             # Lead capture tool + field extraction + validation
│
├── data/
│   └── knowledge_base.json  # Local knowledge base (pricing, plans, policies, FAQs)
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### Prerequisites
- Python **3.9+**
- A free **Groq API key** — sign up at [console.groq.com](https://console.groq.com) 

### Step 1 — Clone the repository
```bash
git clone <https://github.com/kunalkongari/autostream-ai-agent.git>
cd autostream
```

### Step 2 — Create a virtual environment
```bash
python -m venv venv

# Activate it:
# Windows CMD:
venv\Scripts\activate
# Windows PowerShell:
venv\Scripts\Activate.ps1
# Mac/Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Set your Groq API key

**Windows CMD:**
```bash
set GROQ_API_KEY=your-groq-api-key-here
```
**Windows PowerShell:**
```powershell
$env:GROQ_API_KEY="your-groq-api-key-here"
```
**Mac/Linux:**
```bash
export GROQ_API_KEY="your-groq-api-key-here"
```

### Step 5 — Run the agent
```bash
python main.py
```

> ⚠️ You must set the API key and run `python main.py` in the **same terminal window**.

---

## 🔄 How the Agent Works (Conversation Flow)

```
User Message
     │
     ▼
[classify_intent] — regex classifier detects: greeting / product_inquiry / high_intent
     │
     ▼
[router] — reads intent + stage, selects next node
     │
     ├──► [greet]         → warm welcome + company intro
     ├──► [rag]           → retrieve from KB → LLM answers grounded response
     ├──► [collect_lead]  → ask for name → email → platform (one field per turn)
     └──► [capture_lead]  → validate all fields → call mock_lead_capture()
```

**Stage transitions:**
- `idle` → `answering` (after greeting or first product question)
- `answering` → `collecting` (when high intent is detected)
- `collecting` → `capturing` (when all 3 fields are filled)
- `capturing` → `captured` (after tool successfully executes)

---

## 💬 Example Conversation

```
You: Hi!
AutoStream AI: 👋 Hey there! Welcome to AutoStream — your AI-powered video editing co-pilot...

You: What's included in the Pro plan?
AutoStream AI: The Pro Plan at $79/month includes:
  • Unlimited videos per month
  • 4K resolution
  • AI-powered captions
  • Priority 24/7 support
  ...

You: That sounds great, I want to sign up for the Pro plan for my YouTube channel.
AutoStream AI: 🎉 Awesome! I'd love to get you set up. What's your name?

You: Kunal Kongari
AutoStream AI: Nice to meet you, Kunal Kongari! What's your email address?

You: kunal@gmail.com
AutoStream AI: Got it! Which creator platform are you primarily on?

You: YouTube
AutoStream AI: Perfect! Let me get everything set up for you right now... 🚀

==================================================
✅  LEAD CAPTURED SUCCESSFULLY
    Name     : Kunal Kongari
    Email    : kunal@gmail.com
    Platform : Youtube
==================================================

AutoStream AI: ✅ You're all set, kunal Kongari! Our team will reach out at kunal@gmail.com within 24 hours. 🎬
```

---

## 📱 WhatsApp Integration via Webhooks

### Architecture Overview

```
WhatsApp User
      │  sends message
      ▼
Meta WhatsApp Cloud API
      │  POST /webhook  (JSON payload)
      ▼
Your Backend Server  (FastAPI)
      │
      ├── Extract sender_id + message text
      ├── Load AgentState from Redis (keyed by phone number)
      ├── Run graph.invoke(state)
      ├── Save updated state back to Redis
      │
      │  POST reply via WhatsApp API
      ▼
WhatsApp User receives response
```

### Implementation Steps

**1. Register a Webhook** in the [Meta Developer Console](https://developers.facebook.com/):
- Add your server's HTTPS URL as the webhook endpoint
- Meta sends a `GET` to verify, then `POST` for each message

**2. Create a FastAPI webhook endpoint:**

```python
from fastapi import FastAPI, Request
import httpx, json, redis

app = FastAPI()
r = redis.Redis()

@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    body = await request.json()
    msg = body["entry"][0]["changes"][0]["value"]["messages"][0]
    sender_id = msg["from"]
    text = msg["text"]["body"]

    # Load state for this user
    raw = r.get(sender_id)
    state = json.loads(raw) if raw else initial_state()

    # Run agent
    state["messages"].append({"role": "user", "content": text})
    state = graph.invoke(state)

    # Save updated state (1 hour TTL)
    r.set(sender_id, json.dumps(state), ex=3600)

    # Send reply via WhatsApp Cloud API
    async with httpx.AsyncClient() as client:
        await client.post(
            f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages",
            headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"},
            json={
                "messaging_product": "whatsapp",
                "to": sender_id,
                "text": {"body": state["response"]}
            }
        )
    return {"status": "ok"}
```

**3. State Persistence:** Use **Redis** with per-user keys (phone number as key). Each conversation state has a 1-hour TTL. For production, use **DynamoDB** or **PostgreSQL**.

**4. Lead Capture in Production:** Replace `mock_lead_capture()` with a real CRM API call (HubSpot, Salesforce, Pipedrive) and optionally send a Slack/email notification to the sales team.

**5. Deployment:** Host on **Railway**, **Render**, or **AWS Lambda** — Meta requires a public HTTPS URL for webhooks.

> Alternatively, use **Twilio's WhatsApp API** for simpler setup — it provides a sandbox number for testing and a Python SDK.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| Agent Framework | LangGraph (StateGraph) |
| LLM | Llama 3.1 8B via Groq API |
| LLM Interface | LangChain (`langchain-groq`) |
| Knowledge Base | Local JSON file |
| Retrieval | Keyword-based scoring |
| Intent Detection | Regex / rule-based |
| Interface | CLI (terminal) |

---

## 🔮 Future Improvements

- **Semantic RAG** — Replace keyword scoring with sentence-transformers embeddings for better retrieval on complex queries
- **Persistent storage** — Save leads to a database (SQLite locally, PostgreSQL in production)
- **WhatsApp / Telegram deployment** — Add webhook server as described above
- **LLM-based intent fallback** — Use the LLM when regex classifier returns `unknown`
- **Admin dashboard** — View all captured leads in a simple web UI
- **Unit test suite** — Automated tests for intent classifier, RAG retrieval, and tool execution guard
