# VoiceOps 2026: ZentroMart AI Customer Support Agent

[![IIT Guwahati](https://img.shields.io/badge/IIT%20Guwahati-C%26A%20Club-blue.svg)](https://caciitg.com/llm/)
[![Gnani.ai](https://img.shields.io/badge/Powered%20by-Gnani.ai-orange.svg)](https://app.gnani.ai/voice)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An enterprise-grade, grounded voice and conversational support assistant built for the fictional Indian e-commerce retailer **ZentroMart**, developed as the official submission for the **VoiceOps 2026 Capstone Project** ("From Tokens to Transformers: A Course on GenAI x LLM Mastery" by Consulting & Analytics Club, IIT Guwahati × Gnani.ai).

---

## 📑 Table of Contents
- [Executive Overview](#-executive-overview)
- [System Architecture](#-system-architecture)
- [Critical Domain & Dataset Rules](#-critical-domain--dataset-rules)
- [The Four Milestone Steps](#-the-four-milestone-steps)
- [Quantitative Evaluation & Benchmarking](#-quantitative-evaluation--benchmarking)
- [Project Directory Structure](#-project-directory-structure)
- [Installation & Quick Start](#-installation--quick-start)
- [CLI Usage & Batch Evaluation](#-cli-usage--batch-evaluation)
- [Course Curriculum Alignment](#-course-curriculum-alignment)

---

## 🚀 Executive Overview

ZentroMart is a high-volume Indian online store selling groceries, electronics, apparel, and books. Support teams handle thousands of queries daily concerning delivery delays, order cancellations, returns, and refunds. Customers strongly prefer calling over typing.

This project delivers a complete, end-to-end voice-enabled AI support agent that:
1. **Listens (Ears):** Transcribes customer speech across Indian English (`en-IN`), Hindi (`hi-IN`), and Hinglish (`hi-en`) using the official **Gnani STT REST/WebSocket API**.
2. **Reasons (Brain):** Grounds decisions in 21 official policy documents (`kb/*.md`), the product catalog (`data/products.csv`), and pincode serviceability tables (`data/pincodes.csv`). Resolves calendar return windows, business-day refund SLA timelines (accounting for warehouse holidays and Sundays), and enforces an identity verification security gate before disclosing order details.
3. **Speaks (Voice):** Synthesizes natural, culturally tuned audio responses via the **Gnani TTS API** (`model="timbre-v2.5"`, `voice="Kaveri"`), with text normalized for the ear (grouped 8-digit order IDs and spoken currency).

---

## 🏗 System Architecture

```text
                                     +------------------------------------------+
                                     |           CUSTOMER (VOICE / TEXT)        |
                                     +------------------------------------------+
                                            │                            ▲
                         Spoken Audio (.wav)│                            │ Spoken Audio (.wav)
                                            ▼                            │
                       +──────────────────────────+       +──────────────────────────+
                       |        EARS (STT)        |       |        VOICE (TTS)       |
                       | Gnani.ai GnaniSTTClient  |       | Gnani.ai GnaniTTSClient  |
                       | Model: Indic STT (en-IN) |       | Model: timbre-v2.5       |
                       +──────────────────────────+       | Voice: Kaveri            |
                                            │             +──────────────────────────+
                               Transcript   │                            ▲
                                            ▼                            │ Formatted Speech
+────────────────────────────────────────────────────────────────────────────────────────────────────────+
|                                           BRAIN (REASONING & STATE)                                     |
|                                                                                                        |
|  +─────────────────────────+      +──────────────────────────+      +───────────────────────────────+  |
|  |  Identity Gate (IDV-2)  |      |   Hybrid Retrieval RAG   |      |   State Machine & Escalation  |  |
|  | - Withholds order data  | ---> | - Pure-Python BM25       | ---> | - Multi-turn state tracking   |  |
|  | - Phone last 4 / pin    |      | - Metadata precedence    |      | - Repeat contact (ESC-1)      |  |
|  | - Max 1 retry enforced  |      | - Section ID preservation|      | - High-value damage (> ₹10k)  |  |
|  +─────────────────────────+      +──────────────────────────+      +───────────────────────────────+  |
|                 │                               │                                   │                  |
|                 ▼                               ▼                                   ▼                  |
|  +─────────────────────────+      +──────────────────────────+      +───────────────────────────────+  |
|  |    tools.lookup_order   |      |  Calendar & Business Day |      |      tools.create_ticket      |  |
|  | - Gated synthetic data  |      | - Frozen date: 2026-10-20|      | - Handover summary (> 20 ch)  |  |
|  | - orders.csv / customers|      | - Skips Sun & holidays   |      | - Logged to runtime CSV       |  |
|  +─────────────────────────+      +──────────────────────────+      +───────────────────────────────+  |
+────────────────────────────────────────────────────────────────────────────────────────────────────────+
```

---

## 🔒 Critical Domain & Dataset Rules

1. **Frozen Reference Date:**
   - **`2026-10-20` (20 October 2026)** is hardcoded as "today" across the codebase. `datetime.now()` or `date.today()` are strictly avoided for business calculations.
2. **Calendar Days vs. Business Days:**
   - **Calendar Days:** Used for return eligibility windows (`RET-2`, `RET-8`).
   - **Business Days:** Monday to Saturday, **skipping Sundays** and **skipping warehouse holidays** (`data/holidays.csv`: 2026-08-15, 2026-10-02, 2026-10-19, 2026-11-09). Applied to UPI (1–3 days), Card/NEFT (5–7 days), and Net Banking (3–5 days).
3. **Knowledge Base Precedence (KB-00.1):**
   - Hierarchy: **Current Policies (`pol_*.md`) > Agent Guides (`agent_*.md`) > Help Articles (`help_*.md`) > FAQs (`faq_*.md`) > Marketing (`mkt_*.md`)**.
   - Superseded documents (`pol_returns_v3_superseded.md`) are penalized unless a historical date (pre-August 2026) is explicitly queried.
   - Support logs (`data/support_logs.csv`) contain human errors and are **never** used as policy authority.
   - The **FAQ-1 Trap** (stating 5–7 days for all refunds) is prevented: policies govern wallet refunds (instant within 2 hours) and UPI (1–3 business days).
4. **Marketplace Seller Rule (MKS-3):**
   - Products fulfilled by third-party sellers (`fulfilled_by != 'ZentroMart'`) **never** receive ZentroMart Assured warranty, even if the listing shows an assured badge.
5. **Security & Identity Gate (IDV-1 to IDV-4):**
   - Unverified `lookup_order(order_id)` returns zero private details.
   - Customer must provide either the **last 4 digits of registered phone** or **6-digit delivery pincode**.
   - Exactly **one retry** is permitted upon mismatch (`IDV-3`). If it fails a second time, order details are completely refused. Verification secrets are never read aloud.
6. **Voice Layer Rules (Gnani.ai Exclusive):**
   - Strictly uses the official SDK (`pip install gnani-vachana`). No other speech services are permitted.
   - `model="timbre-v2.5"` and `voice="Kaveri"` explicitly specified.
   - Text normalized for the ear: grouped 8-digit order numbers (`4663 7226`) and spoken currency (`49 rupees`).
   - Graceful recovery on silent/unclear audio: `"Sorry, I didn't catch that. Could you say it again?"`

---

## 📦 The Four Milestone Steps

### Step 1: Prompt-Only Baseline (`src/step1_baseline.py`)
- Represents Weeks 1 & 2 curriculum (temperature=0.0, system prompt, zero external documents).
- Identifies as AI assistant on opening (`STY-1`).
- Demonstrates typical hallucinations and policy guessing (e.g. assuming generic 30-day returns and 5–7 day wallet refunds).

### Step 2: Grounded Knowledge Retrieval & RAG (`src/step2_rag.py`)
- Represents Week 3 curriculum.
- Indexes all 21 Markdown documents in `kb/` preserving `[SECTION-ID]` tags (`RET-2`, `REF-2`, `WAR-2`, etc.).
- Indexes product catalog with `products.csv:SKU` reference identifiers.
- Implements pure-Python BM25 + lexical retrieval with precedence weighting and business-day calculation.
- Jumps Must-Include policy compliance from **11.5% to 98.1%** with **100% trap avoidance**.

### Step 3: Multi-Turn Conversational Agent (`src/step3_agent.py`)
- Tracks multi-turn dialogue state and coreference ("Aur refund kitne din mein aayega?").
- Enforces the identity verification gate before revealing order items, status, or tracking.
- Implements tools `lookup_order` and `create_ticket`.
- Executes escalation triggers under `ESC-1` (damage > ₹10,000, repeat contact >= 2 in 7 days, human requested, overdue refunds).

### Step 4: VoiceOps Speech Pipeline (`src/step4_voice.py`)
- Integrates `gnani-vachana` STT and TTS clients reading `GNANI_API_KEY` from the environment.
- Supports batch audio evaluation over entire folders as mandated by the rubric.
- Normalizes text for audio playback and handles unclear speech gracefully.

---

## 📊 Quantitative Evaluation & Benchmarking

Evaluated across all **52 single-turn questions** (`practice/questions.csv`) and **23 multi-turn conversations / 96 turns** (`practice/conversations.csv`):

### 1. Single-Turn Benchmark (52 Questions)
| Metric | Step 1 (Prompt Baseline) | Step 2 (Grounded RAG) | Step 4 (Voice ASR Text) | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Decision Accuracy** | 88.46% | **96.15%** | **96.15%** | +7.69% |
| **Must-Include Keyword Recall** | 11.54% | **98.08%** | **86.54%** | **+86.54% (8.5x)** |
| **Trap Avoidance Rate** | 100.0% | **100.0%** | **100.0%** | 100% compliant |
| **Gold Evidence Section Recall** | 0.00% | **96.15%** | **96.15%** | Full citation |

### 2. Multi-Turn Conversation Benchmark (23 Conversations / 96 Turns)
| Metric | Score Achieved | Capstone Target | Status |
| :--- | :---: | :---: | :---: |
| **Turn Decision Accuracy** | **100.0%** (96/96) | >= 90% | **EXCEEDED** |
| **Must-Include Keyword Match** | **94.79%** (91/96) | >= 85% | **EXCEEDED** |
| **Trap Avoidance Rate** | **100.0%** (96/96) | 100% | **PERFECT** |
| **Tool Call Match Accuracy** | **100.0%** (96/96) | >= 90% | **EXCEEDED** |
| **Identity Verification Compliance** | **100.0%** (Zero leaks) | 100% | **PERFECT** |

---

## 📂 Project Directory Structure

```text
team_name_voiceops/
├── README.md                      # Complete system documentation & quick start
├── .env.example                   # Environment configuration template (no secrets)
├── requirements.txt               # Locked dependencies
├── run.py                         # Unified CLI runner (batch, eval, interactive, steps)
├── REPORT_CONTENT.md              # 4-page capstone report content
├── report.pdf                     # Compiled 4-page official submission PDF
├── src/
│   ├── __init__.py                # Package declaration
│   ├── config.py                  # Fixed date 2026-10-20, business day calculator, paths
│   ├── step1_baseline.py          # Prompt-only LLM baseline
│   ├── step2_rag.py               # Grounded BM25 + catalog RAG with section citations
│   ├── step3_agent.py             # Multi-turn conversational agent with ID gate & tools
│   ├── step4_voice.py             # Official Gnani STT & TTS speech pipeline
│   └── evaluate.py                # Automated benchmark on practice datasets
└── sample_outputs/                # Evaluation reports, JSON logs, and sample run outputs
    ├── step1_vs_step2_comparison.csv
    ├── questions_benchmark.json
    ├── conversations_benchmark.json
    └── voice_batch/
```

---

## ⚙️ Installation & Quick Start

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/your-username/team_name_voiceops.git
cd team_name_voiceops
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your credentials:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GNANI_API_KEY=your_gnani_api_key_here
```

---

## 💻 CLI Usage & Batch Evaluation

### 1. Batch Audio Folder Evaluation (Primary Evaluation Command)
Takes a folder of customer `.wav` or `.mp3` recordings, transcribes with Gnani STT, processes through the Step 3 Agent, and synthesizes spoken replies with Gnani TTS:
```bash
python run.py --input-dir ../test_audio/ --output-dir sample_outputs/voice_batch/
```

### 2. Run the Full Automated Benchmark Suite
```bash
python run.py --eval
```

### 3. Single Query Execution
```bash
python run.py --query "How fast are refunds to the ZentroMart Wallet?"
```

### 4. Interactive Terminal Chat
```bash
python run.py --interactive
```

### 5. Milestone Step Demos
```bash
python run.py --step 1   # Run Step 1 Prompt-only Baseline
python run.py --step 2   # Run Step 2 Grounded RAG
python run.py --step 3   # Run Step 3 Multi-turn Agent with Verification
python run.py --step 4   # Run Step 4 VoiceOps Speech Pipeline
```

---

## 🎓 Course Curriculum Alignment

| Course Week | Curriculum Module | Implemented Component in VoiceOps |
| :--- | :--- | :--- |
| **Week 1** | *Tokens & Embeddings* | Deterministic decoding (`temperature=0.0`), text normalization, and lexical tokenization. |
| **Week 2** | *Prompting & Chatbots* | System prompt engineering, conversation persona (`STY-1`), and evaluation metrics. |
| **Week 3** | *RAG & Tool-Calling Agents* | BM25 chunking over 21 Markdown files, `[SECTION-ID]` preservation, identity gate, and `lookup_order` / `create_ticket` tools. |
| **Week 4** | *VoiceOps Capstone* | Indic speech AI using `gnani-vachana` (`GnaniSTTClient` + `GnaniTTSClient` with `timbre-v2.5` & `Kaveri`). |

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details. Built for IIT Guwahati & Gnani.ai VoiceOps 2026.
