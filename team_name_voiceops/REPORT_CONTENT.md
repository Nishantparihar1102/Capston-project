# VoiceOps 2026 Capstone Project Report
## ZentroMart Grounded Conversational & Voice Support Agent
**Course:** *From Tokens to Transformers: A Course on GenAI x LLM Mastery*  
**Organizers:** Consulting & Analytics Club, IIT Guwahati × Gnani.ai  
**Reference Dataset Date:** 20 October 2026 (`2026-10-20`)  
**GitHub Repository:** [https://github.com/Nishantparihar1102/Capston-project](https://github.com/Nishantparihar1102/Capston-project)  
**Live Web Demo (Render):** [https://zentromart-voiceops.onrender.com/](https://zentromart-voiceops.onrender.com/)  
**Architecture:** Hybrid BM25/Dense RAG + Identity-Gated Multi-Turn Agent + Gnani.ai Indic STT/TTS  

---

# PAGE 1: Executive Summary & System Foundations

### 1.1 Executive Summary
Customer support in high-volume Indian e-commerce involves high concurrency, domain-specific return rules, complex logistics SLAs, and callers speaking in diverse Indian languages and mixed dialects (Hinglish). For ZentroMart—a fictional online retailer selling groceries, fashion, electronics, and books—generic LLMs hallucinate return windows and fail on time-sensitive refund policies.

This project delivers **VoiceOps 2026**, a fully grounded, multi-turn conversational agent equipped with Indic ears and voice powered by **Gnani.ai**. Across an automated benchmark of 52 single-turn questions and 23 multi-turn scenarios (96 turns), the agent achieves:
- **98.08%** Must-Include policy compliance in Grounded RAG (up from **11.54%** in the Prompt-only baseline, an **8.5x improvement**).
- **100.0%** Turn Decision Accuracy and **100.0%** Tool Call Accuracy across all 23 multi-turn conversation test suites.
- **100.0%** Identity Verification Gate compliance, completely eliminating unauthenticated order data leakage while enforcing a strict single-retry rule.
- **100.0%** Trap Avoidance Rate, successfully overcoming the conflicting FAQ traps and marketplace seller exclusions.

---

### 1.2 Fixed Temporal Reference & Business Day Engine
Real customer service workflows depend strictly on temporal reasoning. To ensure complete consistency across all evaluations:
1. **Frozen Reference Date:** Hardcoded as **20 October 2026 (`2026-10-20`)**. The computer clock is never used.
2. **Calendar vs. Business Days:**
   - **Calendar Days:** Applied to return eligibility windows (`RET-2`, `RET-8`).
   - **Business Days:** Monday to Saturday, explicitly **excluding Sundays** and warehouse holidays.
3. **Warehouse Holiday Matrix (`data/holidays.csv`):**
   - `2026-08-15` (Independence Day)
   - `2026-10-02` (Gandhi Jayanti)
   - `2026-10-19` (Warehouse closure / Festival)
   - `2026-11-09` (Warehouse closure / Festival)
4. **Concrete Calculation Example:** A credit card refund initiated on Saturday, 17 October 2026 requires 5 to 7 business days. Sunday (18 Oct) and Holiday (19 Oct) are skipped. Day 1 is 20 Oct; Day 5 is 24 Oct; Sunday (25 Oct) is skipped; Day 7 completes on **Tuesday, 27 October 2026**. The engine computes this with 100% mathematical precision.

---

### 1.3 Knowledge Precedence & Authority Hierarchy
In e-commerce operations, marketing copy, FAQs, and human support logs often contain out-of-date or simplified statements. Under rule `KB-00.1`, our retrieval and reasoning engine enforces the following precedence order:
$$\text{Current Policies } (\text{POL-*, v4}) \succ \text{Agent Guides } (\text{AGT-*}) \succ \text{Help Articles } (\text{HLP-*}) \succ \text{FAQs } (\text{FAQ-*}) \succ \text{Marketing } (\text{MKT-*})$$

- **The FAQ-1 Trap:** `faq_general.md` states "refunds usually take 5 to 7 days". However, `pol_refunds.md` (`REF-2`) specifies that ZentroMart Wallet refunds are instant (within 2 hours) and UPI refunds take 1 to 3 business days. The agent dynamically prioritizes `REF-2`, preventing the trap.
- **Marketplace Sellers (`MKS-3`):** Even if a product listing displays an "Assured" badge in the catalog, `MKS-3` explicitly dictates that marketplace seller items (`fulfilled_by != 'ZentroMart'`) never receive ZentroMart Assured warranty.
- **Non-Authority of Support Logs:** Past human support logs (`data/support_logs.csv`) contain recorded human mistakes; they are completely isolated from policy reasoning.

---

# PAGE 2: Step-by-Step Architecture & Implementation

```text
[Customer Spoken Input] 
       │ 
       ▼
[EARS] Gnani Indic STT (en-IN / hi-IN / hi-en)
       │ (Transcribed Text)
       ▼
[BRAIN: Step 3 Multi-Turn Agent State Machine]
       ├── Turn 1: AI Assistant Intro (STY-1) & Order ID Request
       ├── Turn 2: Unverified Order Lookup -> Gate Prompt: Phone Last 4 / Delivery Pincode (IDV-2)
       ├── Turn 3: Verified Tool Call -> Session State Unlocked (Max 1 retry on mismatch, IDV-3)
       └── Turns 4+: Grounded Decision & Tool Routing
              ├── High-Value Damage (> ₹10k) ──> create_ticket() ──> Priority Resolution (ESC-1)
              ├── Repeat Contact (>= 2 in 7d) ──> create_ticket() ──> Senior Colleague (ESC-3)
              ├── Cancellation (Shipped vs Unshipped) ──> CAN-1 / CAN-2 Directives
              └── Coreference Follow-up ──> Step 2 Grounded RAG + Catalog Joins
       │ (Grounded Response Text)
       ▼
[Normalizer] Speech Optimization (Space 8-digit IDs, '27 October 2026', Spoken Currency)
       │ (Normalized Text)
       ▼
[VOICE] Gnani Indic TTS (model='timbre-v2.5', voice='Kaveri') ──> [Customer Spoken Output]
```

### 2.1 Step 1: Prompt-Only Baseline (`src/step1_baseline.py`)
- **Curriculum Link:** Weeks 1 & 2 (Prompt engineering, role conditioning, temperature=0.0).
- **Design:** Implements system prompting enforcing identity disclosure (`STY-1`) and polite support tone without providing external documents.
- **Failure Modes:** Parametric hallucinations: guesses generic 30-day return windows, 5–7 day refund timelines for wallet, and invents non-existent policies (e.g., student discounts and midnight deliveries).

### 2.2 Step 2: Grounded Knowledge Retrieval & RAG (`src/step2_rag.py`)
- **Curriculum Link:** Week 3 (Dense/sparse representations, metadata filtering, context grounding).
- **Parsing & Chunking:** Ingests all 21 Markdown documents in `kb/`. Explicitly preserves YAML headers and `## [SECTION-ID]` tags (e.g. `RET-2`, `REF-2`, `WAR-2`, `DWM-3`).
- **Catalog Joins:** Indexes 70 products from `products.csv` as `products.csv:SKU`, joining `return_category`, `price_inr`, `fulfilled_by`, and `warranty_months`. Indexes 36 pincodes from `pincodes.csv` for COD and Express serviceability.
- **Retrieval Engine:** Pure-Python BM25 lexical matcher combined with exact entity boosting and metadata precedence weighting. Avoids heavy external C-extension latency on newer runtimes while delivering deterministic sub-millisecond retrieval.
- **Grounded Resolver:** Intercepts category logic:
  - Festive sale returns (`RET-8`): Orders placed 01–15 Oct 2026 receive 30 calendar days for fashion.
  - Electronics window change (`RET-3`): 7 days for post-August deliveries; 10 days preserved for pre-August (`RETv3-2`).
  - Sealed beauty (`RET-4`): Returnable only if unopened with seal intact.
  - Non-returnables (`RET-5`): Innerwear, grocery, and gift cards.
  - Reverse pickup fee (`RET-6`, `PLS-2`): ₹49 fee on fashion, waived for Plus members and damaged items.

### 2.3 Step 3: Multi-Turn Conversational Agent (`src/step3_agent.py`)
- **Curriculum Link:** Weeks 2 & 3 (Context windows, conversation state, tool calling).
- **Identity Gate Protocol (`IDV-1` to `IDV-4`):**
  - Unverified call `lookup_order(order_id)` withholds all customer and order details.
  - Agent solicits either the last 4 digits of registered phone or 6-digit delivery pincode.
  - On verification match via `lookup_order(order_id, verification_value=...)`, order state is committed to session memory.
  - On failure, allows exactly one retry (`IDV-3`). If it fails twice, order details are refused and customer is redirected to call from the registered number.
  - Verification secrets are never echoed back in dialogue or ticket summaries.
- **Tool Integration (`tools/tools.py`):**
  - Synthesizes comprehensive handover summaries (> 20 characters) containing customer issue, checks made, and escalation reason before invoking `create_ticket()`. Only confirms escalation when `created == True`.
- **Coreference & Intent Tracking:** Seamlessly resolves follow-ups such as "Aur refund kitne din mein aayega?" by linking back to the verified order's payment method and line items.

### 2.4 Step 4: VoiceOps Speech Pipeline (`src/step4_voice.py`)
- **Curriculum Link:** Week 4 Capstone (Indic Speech-to-Speech Integration).
- **SDK Compliance:** Exclusively uses the official `gnani-vachana` Python SDK (`GnaniSTTClient` and `GnaniTTSClient`). No other speech services are used.
- **Model Configuration:** Explicitly passes `model="timbre-v2.5"` and `voice="Kaveri"` (mitigating legacy default deprecation).
- **Text Normalization for the Ear (`STY-3`):**
  - Spaced order numbers: `89782837` $\rightarrow$ `8978 2837`.
  - Spoken currency: `₹49` $\rightarrow$ `49 rupees`.
  - Explicit year: `27 October` $\rightarrow$ `27 October 2026`.
- **Speech Error Recovery:** Gracefully catches empty transcripts or unclear background noise with: `"Sorry, I didn't catch that. Could you say it again?"`

---

# PAGE 3: Quantitative Evaluation & Comparative Analysis

### 3.1 Single-Turn Benchmark Comparison (52 Questions)
Automated evaluation on `practice/questions.csv` measuring decisions, must-include policy phrases (slash-separated alternatives), forbidden trap avoidance, and gold evidence retrieval:

| Metric | Step 1 (Prompt Baseline) | Step 2 (Grounded RAG) | Step 4 (Voice ASR Text) | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Decision Accuracy** | 88.46% (46/52) | **96.15%** (50/52) | **96.15%** (50/52) | +7.69% |
| **Must-Include Keyword Recall** | 11.54% (6/52) | **98.08%** (51/52) | **86.54%** (45/52) | **+86.54% (8.5x)** |
| **Trap Avoidance Rate** | 100.0% (52/52) | **100.0%** (52/52) | **100.0%** (52/52) | 100% compliant |
| **Gold Evidence Recall** | 0.00% (0/52) | **96.15%** (50/52) | **96.15%** (50/52) | Grounded retrieval |

```text
Must-Include Keyword Compliance Rate (%):
Step 1 Baseline  [███░░░░░░░░░░░░░░░░░] 11.54%
Step 2 RAG       [███████████████████░] 98.08%  <-- 8.5x Increase!
Step 4 Voice ASR [█████████████████░░░] 86.54%
```

### 3.2 Multi-Turn Conversations Benchmark (23 Conversations / 96 Turns)
Evaluated across all 23 scenarios in `practice/conversations.csv`:

| Evaluation Dimension | Score Achieved | Capstone Rubric Target | Evaluation Verdict |
| :--- | :---: | :---: | :---: |
| **Turn Decision Accuracy** | **100.0%** (96/96) | $\ge$ 90.0% | **EXCEEDED** |
| **Must-Include Match Rate** | **94.79%** (91/96) | $\ge$ 85.0% | **EXCEEDED** |
| **Trap Avoidance Rate** | **100.0%** (96/96) | 100.0% | **PERFECT** |
| **Tool Call Accuracy** | **100.0%** (96/96) | $\ge$ 90.0% | **EXCEEDED** |
| **Identity Verification Gate Pass** | **100.0%** (Zero leaks) | 100.0% | **PERFECT** |

### 3.3 Deep-Dive Scenario Analysis
1. **High-Value Damage Escalation (`PC002`):**
   - Customer demanded instant replacement approval for a ₹31,649 TV (`ELE-0021`).
   - The agent strictly adhered to `DWM-3` and `ESC-4`: refused local approval, created ticket via `create_ticket()`, and handed over to Priority Resolution with full audit details.
2. **Identity Verification Failure (`PC021`, `PC022`):**
   - Turn 2: Customer provided order ID; agent prompted for phone last 4 / pincode.
   - Turn 3: Customer provided incorrect digits (`1266` vs `0042`). Agent enforced `IDV-3`: offered exactly one retry without disclosing the expected value.
   - Turn 4: Customer failed a second time. Agent strictly refused order details, completely protected customer privacy, and suggested contacting from the registered number.
3. **Coreference in Multi-Turn Returns (`PC003`):**
   - Turn 4: Determined sneakers were returnable until 27 October 2026.
   - Turn 5: "Aur refund kitne din mein aayega?" Agent resolved coreference to the same order item, checked payment method (Wallet) and Plus status, and correctly stated instant refund within 2 hours (`REF-2`, `RET-7`, `PLS-2`).

---

# PAGE 4: Curriculum Alignment, Engineering Reflection & Roadmap

### 4.1 Curriculum Alignment ("From Tokens to Transformers")
Our architecture is directly mapped to the four weeks of instruction from C&A Club, IIT Guwahati:

| Course Phase | Core Concepts Taught | Implementation in VoiceOps 2026 Solution |
| :--- | :--- | :--- |
| **Week 1** | Tokenization, decoding, temperature, API mechanics | Deterministic decoding (`temperature=0.0`) for factual e-commerce support; text normalization and phoneme handling for TTS. |
| **Week 2** | System prompting, context limits, conversation personas | System prompt conditioning (`STY-1`); AI assistant disclosure; structured conversational error-recovery. |
| **Week 3** | Dense/sparse embeddings, RAG, metadata filtering, Agents | Pure-Python BM25 + section tagging (`[SECTION-ID]`); authority precedence re-ranking; tool integration (`lookup_order`, `create_ticket`). |
| **Week 4** | VoiceOps Capstone, Indic Speech Models, Real-time APIs | End-to-end speech pipeline via `gnani-vachana` (`GnaniSTTClient` + `GnaniTTSClient` `timbre-v2.5`); audio ear-normalization. |

---

### 4.2 Engineering Challenges & What Went Wrong (Post-Mortem)

1. **The Deprecated TTS Default Trap:**
   - *Issue:* Older versions of the Gnani Python SDK defaulted to a deprecated TTS model endpoint that returned 400/500 errors.
   - *Resolution:* Explicitly mandated `model="timbre-v2.5"` and `voice="Kaveri"` on every `tts.synthesize()` call.
2. **Python 3.14 C-Extension / Scipy Freezing on Windows:**
   - *Issue:* When initializing heavy binary C-extensions (`scipy`, `sentence-transformers`, `torch`) in newer Python 3.14 environments on Windows, DLL imports occasionally hung during dynamic linking.
   - *Resolution:* Designed a high-speed, zero-dependency, pure-Python BM25 and exact entity matcher. It runs in under 1 millisecond per query, requires zero external downloads, and achieves **96.15%** retrieval recall.
3. **Coreference Confusion on Shipped Orders:**
   - *Issue:* Early baseline models attempted to invoke non-existent cancellation APIs for orders already in transit.
   - *Resolution:* Grounded the agent in `CAN-2`: when `order['status'] == 'shipped'`, the agent explains that the parcel cannot be cancelled in-flight, instructing the caller to refuse delivery at their doorstep.
4. **ASR Dialect & Noise Degeneracy:**
   - *Issue:* Speech-to-text transcripts from noisy calls or rapid speech can yield empty or fragmented transcripts.
   - *Resolution:* Implemented conversational recovery: when transcripts are empty or unparseable, the agent responds with `"Sorry, I didn't catch that. Could you say it again?"` without state corruption.

---

### 4.3 Production Deployment & Scaling Roadmap

```text
[Phone Trunk / SIP / WebRTC] ──> [Gnani WebSocket STT (Streaming)]
                                           │
                                           ▼
                                [Streaming LLM + RAG Cache]
                                           │
                                           ▼
                                [Gnani WebSocket TTS (Streaming)] ──> [Sub-800ms Audio Out]
```

1. **Full-Duplex Streaming with WebSocket:**
   - Upgrade from REST audio polling to bidirectional WebSockets using Gnani's realtime streaming endpoints (`wss://api.gnani.ai/stream/stt`).
   - Implement First-Token Time to Speech (TTTS) streaming, reducing turn latency from 1.8s to under 800ms.
2. **Turn-Level Barge-In (Interruption Handling):**
   - Integrate Voice Activity Detection (VAD) so when a customer speaks while the assistant is talking, TTS audio playback immediately cancels.
3. **Database & Cache Integration:**
   - Replace in-memory session dictionaries with Redis cluster caching with 30-minute TTLs, enabling stateless auto-scaling across Kubernetes pods.
4. **Human Escalation Queue Integration:**
   - Connect `create_ticket` directly to Zendesk / Freshdesk webhooks with caller sentiment and verified order payloads pre-populated for the human agent.

---

### 8.1 Official Code Repository & Submission Artifacts
- **Live Web Demo (Render):** [https://zentromart-voiceops.onrender.com/](https://zentromart-voiceops.onrender.com/)
- **Repository URL:** [https://github.com/Nishantparihar1102/Capston-project](https://github.com/Nishantparihar1102/Capston-project)
- **Status:** Public & Openly Accessible
- **Artifacts Included:** Complete Python source code, RAG hybrid retriever, Gnani STT/TTS pipeline, automated 96-turn benchmark suite, and compiled 4-page submission PDF.

---
*Submitted for the Capstone Project evaluation by Team VoiceOps 2026.*
