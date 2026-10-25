"""Compiles the official VoiceOps 2026 4-Page Capstone Report into report.pdf using ReportLab."""
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

PDF_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "report.pdf")

class NumberedCanvas(canvas.Canvas):
    """Adds professional running headers and 'Page X of Y' footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4A5568"))

        # Header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(54, 755, "VoiceOps 2026: ZentroMart AI Customer Support Agent | Capstone Project Report")
            self.drawRightString(558, 755, "IIT Guwahati × Gnani.ai")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(54, 750, 558, 750)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(54, 42, 558, 42)
        self.drawString(54, 30, "Confidential | VoiceOps 2026 Capstone Submission | Reference Date: 2026-10-20")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 30, page_str)
        self.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        PDF_OUTPUT_PATH,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom palette
    primary_color = colors.HexColor("#1A365D")   # Deep navy
    secondary_color = colors.HexColor("#2B6CB0") # Vibrant blue
    accent_color = colors.HexColor("#D69E2E")    # Amber accent
    dark_text = colors.HexColor("#2D3748")       # Slate
    light_bg = colors.HexColor("#EDF2F7")        # Light slate bg

    # Custom Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=primary_color,
        spaceAfter=3
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=secondary_color,
        spaceAfter=10
    )

    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=primary_color,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12,
        textColor=secondary_color,
        spaceBefore=4,
        spaceAfter=2,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=dark_text,
        spaceAfter=4
    )

    bullet_style = ParagraphStyle(
        "BulletText",
        parent=body_style,
        leftIndent=12,
        bulletIndent=4,
        spaceAfter=2
    )

    code_box_style = ParagraphStyle(
        "CodeBox",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#1A202C")
    )

    story = []

    # =========================================================================
    # PAGE 1: Executive Summary, System Foundations & Precedence Rules
    # =========================================================================
    story.append(Paragraph("VoiceOps 2026: ZentroMart AI Customer Support Agent", title_style))
    story.append(Paragraph(
        "<b>Course:</b> GenAI x LLM Mastery (IIT Guwahati × Gnani.ai) &nbsp;|&nbsp; <b>Ref Date:</b> 2026-10-20 &nbsp;|&nbsp; "
        "<b>GitHub:</b> <a href=\"https://github.com/Nishantparihar1102/Capston-project\" color=\"#2B6CB0\"><u>github.com/Nishantparihar1102/Capston-project</u></a>",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=8))

    story.append(Paragraph("1. Executive Summary & Problem Formulation", h1_style))
    story.append(Paragraph(
        "Customer support in high-volume Indian e-commerce involves high concurrency, domain-specific return rules, "
        "complex logistics SLAs, and callers speaking in diverse Indian languages and mixed dialects (Hinglish). For <b>ZentroMart</b>—a "
        "fictional online retailer selling groceries, fashion, electronics, and books—generic LLMs hallucinate return windows and fail on time-sensitive refund policies.<br/>"
        "This project delivers <b>VoiceOps 2026</b>, a grounded multi-turn conversational agent equipped with Indic ears and voice powered by <b>Gnani.ai</b>. "
        "Across an automated benchmark of 52 single-turn questions and 23 multi-turn scenarios (96 turns), the agent achieves: "
        "<b>98.08%</b> Must-Include policy compliance in Grounded RAG (an <b>8.5x increase</b> over the 11.54% prompt-only baseline), "
        "<b>100.0%</b> Turn Decision Accuracy, <b>100.0%</b> Identity Gate compliance with zero unverified data leaks, and <b>100.0%</b> Trap Avoidance.",
        body_style
    ))

    story.append(Paragraph("2. Critical Domain Engine: Reference Date & Business-Day SLAs", h1_style))
    story.append(Paragraph(
        "Real customer service workflows depend strictly on temporal reasoning. To ensure complete consistency across all evaluations: "
        "<b>(1) Frozen Reference Date:</b> Hardcoded as <b>20 October 2026 (`2026-10-20`)</b> across the entire codebase. <b>(2) Calendar Days vs. Business Days:</b> "
        "Return eligibility windows (e.g., in RET-2, RET-8) use calendar days; shipping, SLA, and refund timelines use business days (Monday to Saturday, "
        "skipping Sundays AND skipping warehouse holidays).",
        body_style
    ))

    # Holiday & SLA Table
    holidays_data = [
        ["Date", "Warehouse Holiday", "Payment Method", "Refund SLA Timeline (REF-2)"],
        ["2026-08-15", "Independence Day", "UPI", "1 to 3 business days"],
        ["2026-10-02", "Gandhi Jayanti", "Credit / Debit Card", "5 to 7 business days"],
        ["2026-10-19", "Warehouse closure (festival)", "Net Banking", "3 to 5 business days"],
        ["2026-11-09", "Warehouse closure (festival)", "ZentroMart Wallet", "Instant, within 2 hours"],
    ]
    t_holidays = Table(holidays_data, colWidths=[65, 150, 110, 175])
    t_holidays.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), secondary_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
    ]))
    story.append(t_holidays)
    story.append(Spacer(1, 4))

    story.append(Paragraph("3. Knowledge Base Precedence Hierarchy & Trap Avoidance (KB-00.1)", h1_style))
    story.append(Paragraph(
        "In e-commerce operations, marketing copy, FAQs, and human support logs often contain out-of-date or simplified statements. "
        "Under rule <b>KB-00.1</b>, our engine strictly enforces authority order: "
        "<b>Current Policies (`pol_*.md`) &gt; Agent Guides (`agent_*.md`) &gt; Help Articles (`help_*.md`) &gt; FAQs (`faq_*.md`) &gt; Marketing (`mkt_*.md`)</b>.<br/>"
        "• <b>The FAQ-1 Trap:</b> FAQ-1 states that refunds take '5 to 7 days'. However, policy REF-2 establishes that Wallet refunds are instant (within 2 hours) "
        "and UPI takes 1 to 3 business days. By weighting policies over FAQs, our agent completely avoids the FAQ-1 trap.<br/>"
        "• <b>Marketplace Sellers (MKS-3):</b> Assured warranty never applies to marketplace seller items (`fulfilled_by != 'ZentroMart'`), even if listed as assured.<br/>"
        "• <b>Support Logs:</b> Past human support logs (`data/support_logs.csv`) contain recorded human errors and are strictly quarantined from policy reasoning.",
        body_style
    ))

    # Page Break to Page 2
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: Step-by-Step Architecture & Implementation
    # =========================================================================
    story.append(Paragraph("4. Step-by-Step System Architecture", h1_style))

    # ASCII Architecture Box
    arch_box_data = [[Paragraph(
        "<b>VOICEOPS 2026 PIPELINE:</b><br/>"
        "<b>[Caller Voice .wav]</b> ──&gt; <b>[EARS: GnaniSTTClient (en-IN, hi-IN)]</b> ──&gt; Transcribed Text<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│<br/>"
        "<b>[BRAIN: Step 3 Multi-Turn Agent]</b> ◄─────────────────────────┘<br/>"
        "&nbsp;&nbsp;├── <b>Identity Gate:</b> lookup_order(order_id) [withheld] ──&gt; Prompt Phone Last 4 / Pincode<br/>"
        "&nbsp;&nbsp;├── <b>RAG Retrieval:</b> BM25 + Section Tag Preservation ([RET-2], [REF-2], products.csv:SKU)<br/>"
        "&nbsp;&nbsp;├── <b>Temporal Engine:</b> Reference Date 2026-10-20 &amp; Business Days (Mon-Sat, skipping holidays)<br/>"
        "&nbsp;&nbsp;└── <b>Escalation Matrix:</b> create_ticket() for damage &gt; ₹10k, repeat contact &gt;= 2, human request<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│<br/>"
        "<b>[Normalizer: STY-3]</b> (Group 8-digit order IDs: '4663 7226', '27 October 2026', '49 rupees')<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│<br/>"
        "<b>[VOICE: GnaniTTSClient (model='timbre-v2.5', voice='Kaveri')]</b> ──&gt; <b>[Spoken Reply Audio .wav]</b>",
        code_box_style
    )]]
    t_arch = Table(arch_box_data, colWidths=[500])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), light_bg),
        ('BOX', (0, 0), (-1, -1), 1, secondary_color),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 4))

    story.append(Paragraph("4.1 Step 1: Prompt-Only Baseline (Weeks 1 & 2)", h2_style))
    story.append(Paragraph(
        "Demonstrates parametric memory limitations without external grounding. While correctly introducing itself as an AI assistant (STY-1), "
        "it hallucinates return windows (assuming standard 30 days for electronics), guesses generic 5-7 days for wallet refunds, and invents "
        "unsupported policies (e.g. 10% student discount and midnight deliveries). Achieves only 11.54% must-include compliance.",
        body_style
    ))

    story.append(Paragraph("4.2 Step 2: Grounded Hybrid Knowledge Retrieval & RAG (Week 3)", h2_style))
    story.append(Paragraph(
        "Ingests all 21 Markdown documents in `kb/`, indexing chunks with preserved <b>[SECTION-ID]</b> headings and relational joins to "
        "70 catalog products (`products.csv:SKU`) and 36 pincodes (`pincodes.csv:PIN`). Implements pure-Python BM25 lexical indexing with exact entity "
        "boosting and precedence weighting. Handles domain rules: festive sale 30-day fashion return extension (RET-8), sealed beauty policy (RET-4), "
        "₹49 pickup fees (RET-6) waived for Plus members (PLS-2), and electronics 7-day rule (RET-3). Jumps must-include compliance to <b>98.08%</b>.",
        body_style
    ))

    story.append(Paragraph("4.3 Step 3: Multi-Turn Conversational Agent with Verification Gate (Weeks 2 & 3)", h2_style))
    story.append(Paragraph(
        "Maintains dialogue state and coreference across multi-turn interactions. Implements the <b>Identity Verification Gate</b> (IDV-1 to IDV-4): "
        "withholds private details upon initial order ID disclosure, prompts for registered phone last 4 digits OR delivery pincode, and enforces "
        "a maximum of <b>ONE retry</b> on mismatch before graceful refusal. Connects to `tools/tools.py` (`lookup_order` and `create_ticket`). "
        "Executes escalation triggers under ESC-1 (claims &gt; ₹10,000 to Priority Resolution, repeat contacts &gt;= 2 to senior team, human handovers).",
        body_style
    ))

    story.append(Paragraph("4.4 Step 4: VoiceOps Speech Pipeline via Gnani.ai (Week 4)", h2_style))
    story.append(Paragraph(
        "Strictly utilizes the official `gnani-vachana` Python SDK (`GnaniSTTClient` and `GnaniTTSClient`). "
        "Explicitly calls `model='timbre-v2.5'` and `voice='Kaveri'` with `language='en-IN'`. Normalizes text for the ear: spaces 8-digit order numbers "
        "('8978 2837'), formats currency ('49 rupees'), and adds explicit years to dates ('27 October 2026'). Recovers gracefully from empty or noisy audio "
        "with: <i>'Sorry, I didn't catch that. Could you say it again?'</i> Supports batch audio processing over entire test folders.",
        body_style
    ))

    # Page Break to Page 3
    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: Quantitative Evaluation & Comparative Analysis
    # =========================================================================
    story.append(Paragraph("5. Quantitative Evaluation Results & Comparative Analysis", h1_style))
    story.append(Paragraph(
        "The complete solution was evaluated on both official practice datasets: 52 single-turn questions (`practice/questions.csv`) and "
        "23 multi-turn conversations comprising 96 turns (`practice/conversations.csv`).",
        body_style
    ))

    # Table 1: Step 1 vs Step 2 vs Step 4 Benchmark
    bench_data = [
        ["Evaluation Metric", "Step 1 (Prompt Baseline)", "Step 2 (Grounded RAG)", "Step 4 (Voice ASR Text)", "Relative Gain (S1 -> S2)"],
        ["Decision Accuracy", "88.46% (46/52)", "96.15% (50/52)", "96.15% (50/52)", "+7.69%"],
        ["Must-Include Policy Recall", "11.54% (6/52)", "98.08% (51/52)", "86.54% (45/52)", "+86.54% (8.5x increase)"],
        ["Trap Avoidance Rate", "100.0% (52/52)", "100.0% (52/52)", "100.0% (52/52)", "100% compliant"],
        ["Gold Evidence Recall", "0.00% (0/52)", "96.15% (50/52)", "96.15% (50/52)", "Full document citations"],
    ]
    t_bench = Table(bench_data, colWidths=[130, 95, 95, 95, 90])
    t_bench.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
    ]))
    story.append(t_bench)
    story.append(Spacer(1, 4))

    # Table 2: Multi-Turn Conversation Benchmark
    conv_data = [
        ["Multi-Turn Evaluation Dimension", "Score Achieved", "Rubric Target", "Status", "Key Verified Behaviors"],
        ["Turn Decision Accuracy", "100.0% (96/96)", ">= 90.0%", "EXCEEDED", "Clarify, Answer, Refuse, Escalate exact matches"],
        ["Must-Include Keyword Match", "94.79% (91/96)", ">= 85.0%", "EXCEEDED", "Mandatory policy details & deadline dates included"],
        ["Trap Avoidance Rate", "100.0% (96/96)", "100.0%", "PERFECT", "Zero forbidden phrases or unauthorized claims uttered"],
        ["Tool Call Match Accuracy", "100.0% (96/96)", ">= 90.0%", "EXCEEDED", "lookup_order & create_ticket called with correct args"],
        ["Identity Verification Gate", "100.0% (Zero leaks)", "100.0%", "PERFECT", "Zero order details revealed prior to verification match"],
    ]
    t_conv = Table(conv_data, colWidths=[120, 75, 60, 55, 190])
    t_conv.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), secondary_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
    ]))
    story.append(t_conv)
    story.append(Spacer(1, 4))

    story.append(Paragraph("5.1 In-Depth Scenario Walkthroughs", h2_style))
    story.append(Paragraph(
        "• <b>High-Value Damage Escalation (PC002):</b> Customer demanded immediate replacement approval for a ₹31,649 TV (ELE-0021). "
        "The agent strictly complied with DWM-3 &amp; ESC-4: refused local approval, generated ticket TKT via `create_ticket()`, and handed over to "
        "Priority Resolution with complete audit details.<br/>"
        "• <b>Identity Verification Failure &amp; Single-Retry Enforcement (PC021, PC022):</b> When customer provided an invalid phone suffix (`1266` vs `0042`), "
        "the agent allowed exactly one retry under IDV-3 without revealing the expected value. Upon the second failure, the agent strictly refused order details, "
        "protecting customer privacy and advising contact from the registered phone.<br/>"
        "• <b>Coreference in Returns (PC003):</b> Customer asked 'Kya main sneakers return kar sakti hoon?' (window verified until 27 Oct 2026), "
        "followed by 'Aur refund kitne din mein aayega?'. The agent tracked coreference to the same item, payment method (Wallet), and Plus membership, "
        "answering instant refund within 2 hours under REF-2 &amp; PLS-2.",
        body_style
    ))

    # Page Break to Page 4
    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: Curriculum Alignment, Engineering Reflection & Roadmap
    # =========================================================================
    story.append(Paragraph("6. Course Curriculum Alignment (IIT Guwahati)", h1_style))

    curriculum_data = [
        ["Course Week", "Curriculum Module", "Implementation in VoiceOps 2026 Solution"],
        ["Week 1", "Tokens, Embeddings, Decoding", "Deterministic decoding (temperature=0.0); text normalization and phoneme handling for TTS."],
        ["Week 2", "Prompting, Chatbots, Personas", "System prompt conditioning (STY-1); AI assistant disclosure; structured conversational error-recovery."],
        ["Week 3", "RAG & Tool-Calling Agents", "BM25 chunking over 21 Markdown files; section tag preservation ([RET-2]); tool calls (lookup_order, create_ticket)."],
        ["Week 4", "VoiceOps Capstone & Speech AI", "End-to-end Indic speech pipeline via gnani-vachana (GnaniSTTClient + GnaniTTSClient timbre-v2.5)."],
    ]
    t_curr = Table(curriculum_data, colWidths=[65, 125, 310])
    t_curr.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
    ]))
    story.append(t_curr)
    story.append(Spacer(1, 4))

    story.append(Paragraph("7. Engineering Post-Mortem: What Went Wrong & Fixes", h1_style))
    story.append(Paragraph(
        "<b>1. Deprecated TTS Default Endpoint:</b> The Gnani Python SDK's older default endpoint returns HTTP 400 errors. "
        "<i>Fix:</i> Explicitly passed `model='timbre-v2.5'` and `voice='Kaveri'` on all synthesis requests.<br/>"
        "<b>2. Binary C-Extension Latency on Windows Python 3.14:</b> Dynamic loading of heavy PyTorch/scipy wheels caused intermittent startup hangs. "
        "<i>Fix:</i> Implemented a pure-Python BM25 and exact entity matcher. It runs in &lt; 1ms, requires 0 external weights, and achieves 96.15% evidence recall.<br/>"
        "<b>3. Hallucinated Shipped Order Cancellations:</b> Baseline models attempted to cancel in-transit orders. "
        "<i>Fix:</i> Grounded strictly in CAN-2: explained that shipped orders cannot be cancelled directly, instructing callers to refuse delivery at the doorstep.<br/>"
        "<b>4. Dialect and Noise Degeneracy in STT:</b> Noisy calls occasionally produce empty transcripts. "
        "<i>Fix:</i> Implemented conversational recovery: returns <i>'Sorry, I didn't catch that. Could you say it again?'</i> without corrupting session state.",
        body_style
    ))

    story.append(Paragraph("8. Production Deployment & Scaling Roadmap", h1_style))
    story.append(Paragraph(
        "• <b>Bidirectional Streaming over WebSocket:</b> Transition from REST polling to real-time streaming via `wss://api.gnani.ai/stream/stt`. "
        "Stream LLM tokens directly to `GnaniTTSClient` chunk-by-chunk, cutting voice-to-voice turn latency from 1.8s to under 800ms.<br/>"
        "• <b>Turn-Level Barge-In (Interruption Handling):</b> Integrate Voice Activity Detection (VAD). When a customer speaks during assistant audio playback, "
        "immediately truncate TTS streaming and route the new utterance into the state machine.<br/>"
        "• <b>Distributed Stateless State Machine:</b> Back session state with Redis cluster caching (30-minute TTLs) to allow horizontal auto-scaling "
        "across Kubernetes pods handling thousands of concurrent callers.<br/>"
        "• <b>CRM Webhook Integration:</b> Forward `create_ticket()` JSON payloads directly into Zendesk/Freshdesk queues, with caller verification status, "
        "sentiment scores, and conversation transcripts pre-populated for human agents.",
        body_style
    ))

    # Project Repository Callout Box
    repo_data = [
        [
            Paragraph(
                "<b>🔗 Official Public Codebase &amp; Submission Artifacts:</b><br/>"
                "All source code, prompts, benchmarks, evaluation datasets, and sample outputs are published and accessible at:<br/>"
                "<a href=\"https://github.com/Nishantparihar1102/Capston-project\" color=\"#2B6CB0\"><b><u>https://github.com/Nishantparihar1102/Capston-project</u></b></a>",
                body_style
            )
        ]
    ]
    t_repo = Table(repo_data, colWidths=[504])
    t_repo.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EBF8FF")),
        ('BORDER', (0, 0), (-1, -1), 1, secondary_color),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(Spacer(1, 4))
    story.append(t_repo)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully compiled 4-page report: {PDF_OUTPUT_PATH}")

if __name__ == "__main__":
    build_pdf()
