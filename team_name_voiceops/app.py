"""Live Web Application Server for VoiceOps 2026 (ZentroMart Support Agent).

Provides:
1. Browser Mic Recording & Audio Upload (Web Audio API)
2. Live Multi-Turn Conversational Chat with Speech Output
3. Visual Identity Verification Status Gate
4. Policy Evidence & Tool Execution Badges
5. Preset 1-Click Practice Scenarios
"""
import os
import sys
import json
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from src.config import REFERENCE_DATE_STR, GNANI_API_KEY, GEMINI_API_KEY
from src.step3_agent import Step3Agent
from src.step4_voice import VoiceOpsAgent, normalize_text_for_ear

agent_brain = Step3Agent()
voice_agent = VoiceOpsAgent()

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ZentroMart VoiceOps 2026 | Live AI Agent</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(23, 32, 54, 0.7);
      --card-border: rgba(255, 255, 255, 0.08);
      --primary: #3b82f6;
      --primary-glow: rgba(59, 130, 246, 0.3);
      --accent: #f59e0b;
      --success: #10b981;
      --danger: #ef4444;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Outfit', sans-serif;
      background: radial-gradient(circle at top right, #1e1b4b, var(--bg) 60%);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    header {
      padding: 16px 28px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--card-border);
      backdrop-filter: blur(12px);
      background: rgba(11, 15, 25, 0.8);
      position: sticky;
      top: 0;
      z-index: 100;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .brand-logo {
      width: 36px;
      height: 36px;
      background: linear-gradient(135deg, #3b82f6, #ec4899);
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      font-size: 20px;
    }

    .badge-ref {
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      padding: 4px 10px;
      background: rgba(245, 158, 11, 0.15);
      color: #fbbf24;
      border: 1px solid rgba(245, 158, 11, 0.3);
      border-radius: 20px;
    }

    .main-container {
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 20px;
      padding: 20px;
      max-width: 1400px;
      width: 100%;
      margin: 0 auto;
      flex: 1;
    }

    /* Sidebar controls */
    .sidebar {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .panel {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 18px;
      backdrop-filter: blur(16px);
    }

    .panel h3 {
      font-size: 14px;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 12px;
    }

    .status-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 8px 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      font-size: 13px;
    }

    .preset-btn {
      width: 100%;
      text-align: left;
      padding: 10px 12px;
      margin-bottom: 8px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--card-border);
      color: var(--text);
      border-radius: 8px;
      font-family: inherit;
      font-size: 12px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .preset-btn:hover {
      background: rgba(59, 130, 246, 0.15);
      border-color: var(--primary);
      transform: translateX(3px);
    }

    /* Chat Area */
    .chat-panel {
      display: flex;
      flex-direction: column;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 20px;
      backdrop-filter: blur(16px);
      overflow: hidden;
      height: calc(100vh - 120px);
    }

    .chat-history {
      flex: 1;
      padding: 24px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    .msg {
      max-width: 75%;
      padding: 14px 18px;
      border-radius: 16px;
      font-size: 14.5px;
      line-height: 1.5;
      position: relative;
    }

    .msg.user {
      align-self: flex-end;
      background: linear-gradient(135deg, #2563eb, #1d4ed8);
      color: white;
      border-bottom-right-radius: 4px;
    }

    .msg.agent {
      align-self: flex-start;
      background: rgba(30, 41, 59, 0.85);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-bottom-left-radius: 4px;
    }

    .meta-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }

    .evidence-tag {
      font-family: 'JetBrains Mono', monospace;
      font-size: 10px;
      padding: 2px 8px;
      background: rgba(59, 130, 246, 0.15);
      border: 1px solid rgba(59, 130, 246, 0.3);
      color: #93c5fd;
      border-radius: 12px;
    }

    .tool-tag {
      font-family: 'JetBrains Mono', monospace;
      font-size: 10px;
      padding: 2px 8px;
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: #6ee7b7;
      border-radius: 12px;
    }

    /* Input Controls */
    .input-bar {
      padding: 16px 20px;
      background: rgba(15, 23, 42, 0.9);
      border-top: 1px solid var(--card-border);
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .input-bar input {
      flex: 1;
      padding: 12px 18px;
      background: rgba(30, 41, 59, 0.8);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      color: white;
      font-family: inherit;
      font-size: 14px;
      outline: none;
    }
    .input-bar input:focus {
      border-color: var(--primary);
    }

    .btn-action {
      padding: 12px 20px;
      background: var(--primary);
      color: white;
      border: none;
      border-radius: 12px;
      font-weight: 600;
      cursor: pointer;
      font-family: inherit;
      transition: all 0.2s;
    }
    .btn-action:hover {
      background: #2563eb;
    }

    .btn-mic {
      width: 46px;
      height: 46px;
      border-radius: 50%;
      background: rgba(239, 68, 68, 0.15);
      border: 1px solid rgba(239, 68, 68, 0.4);
      color: #f87171;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      font-size: 18px;
      transition: all 0.2s;
    }
    .btn-mic.recording {
      background: var(--danger);
      color: white;
      box-shadow: 0 0 16px rgba(239, 68, 68, 0.6);
      animation: pulse 1s infinite alternate;
    }

    @keyframes pulse {
      from { transform: scale(1); }
      to { transform: scale(1.1); }
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="brand-logo">Z</div>
      <div>
        <h1 style="font-size: 18px; font-weight: 700;">ZentroMart VoiceOps 2026</h1>
        <p style="font-size: 11px; color: var(--text-muted);">IIT Guwahati × Gnani.ai Capstone Project</p>
      </div>
    </div>
    <div style="display: flex; gap: 10px; align-items: center;">
      <span class="badge-ref">REF DATE: 2026-10-20</span>
      <span id="badge-auth" style="font-size: 11px; padding: 4px 10px; background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 20px;">🔒 UNVERIFIED</span>
    </div>
  </header>

  <div class="main-container">
    <div class="sidebar">
      <div class="panel">
        <h3>Session State</h3>
        <div class="status-item"><span>Order ID:</span> <span id="lbl-oid" style="font-family: monospace;">None</span></div>
        <div class="status-item"><span>Customer:</span> <span id="lbl-cust">-</span></div>
        <div class="status-item"><span>Identity Gate:</span> <span id="lbl-gate" style="color: #f87171;">Locked (IDV-2)</span></div>
        <div class="status-item"><span>Speech SDK:</span> <span style="color: #10b981;">Gnani timbre-v2.5</span></div>
      </div>

      <div class="panel">
        <h3>Quick Test Scenarios</h3>
        <button class="preset-btn" onclick="sendPreset('Hello, mujhe order ke baare mein help chahiye.')">1. Opening Greeting (STY-1)</button>
        <button class="preset-btn" onclick="sendPreset('Order number hai 8978 2837.')">2. Provide Order ID (IDV-2)</button>
        <button class="preset-btn" onclick="sendPreset('Last four digits 4533 hain.')">3. Verify Phone Suffix (Match)</button>
        <button class="preset-btn" onclick="sendPreset('Maine Zentro Vista 43-inch 4K Smart TV damage report kiya tha, replacement approve kar do na?')">4. High Value Damage (&gt; ₹10k)</button>
        <button class="preset-btn" onclick="sendPreset('How fast are refunds to the ZentroMart Wallet?')">5. Wallet Refund (REF-2 Trap)</button>
        <button class="preset-btn" onclick="sendPreset('Actually, can I speak to a person?')">6. Human Handover (ESC-1)</button>
      </div>
    </div>

    <div class="chat-panel">
      <div class="chat-history" id="chat">
        <div class="msg agent">
          Hello! I am ZentroMart's AI support assistant. How can I help you today? Please share your 8-digit order number if you are inquiring about an order.
        </div>
      </div>

      <div class="input-bar">
        <button class="btn-mic" id="mic-btn" onclick="toggleMic()" title="Hold to speak">🎙️</button>
        <input type="text" id="user-input" placeholder="Type in English, Hindi, or Hinglish (or click mic to talk)..." onkeydown="if(event.key==='Enter') sendText()">
        <button class="btn-action" onclick="sendText()">Send</button>
      </div>
    </div>
  </div>

  <script>
    let isRecording = false;
    let mediaRecorder = null;
    let audioChunks = [];

    async function sendText() {
      const input = document.getElementById('user-input');
      const text = input.value.trim();
      if (!text) return;

      appendMsg(text, 'user');
      input.value = '';

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text })
        });
        const data = await res.json();
        handleAgentResponse(data);
      } catch (e) {
        appendMsg("Error connecting to server.", 'agent');
      }
    }

    function sendPreset(text) {
      document.getElementById('user-input').value = text;
      sendText();
    }

    function appendMsg(text, sender, meta = null) {
      const chat = document.getElementById('chat');
      const div = document.createElement('div');
      div.className = `msg ${sender}`;
      div.innerText = text;

      if (meta && (meta.gold_evidence?.length || meta.tool_calls?.length)) {
        const tagBox = document.createElement('div');
        tagBox.className = 'meta-tags';
        (meta.gold_evidence || []).forEach(e => {
          const t = document.createElement('span');
          t.className = 'evidence-tag';
          t.innerText = `Evidence: ${e}`;
          tagBox.appendChild(t);
        });
        (meta.tool_calls || []).forEach(tc => {
          const t = document.createElement('span');
          t.className = 'tool-tag';
          t.innerText = `Tool: ${tc}`;
          tagBox.appendChild(t);
        });
        div.appendChild(tagBox);
      }

      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
    }

    function handleAgentResponse(data) {
      appendMsg(data.reply, 'agent', data);

      if (data.verified) {
        document.getElementById('badge-auth').innerText = '🔓 VERIFIED';
        document.getElementById('badge-auth').style.background = 'rgba(16, 185, 129, 0.15)';
        document.getElementById('badge-auth').style.color = '#10b981';
        document.getElementById('badge-auth').style.borderColor = 'rgba(16, 185, 129, 0.3)';
        document.getElementById('lbl-gate').innerText = 'Verified (Unlocked)';
        document.getElementById('lbl-gate').style.color = '#10b981';
      }

      if (data.order_id) {
        document.getElementById('lbl-oid').innerText = data.order_id;
      }
      if (data.customer_name) {
        document.getElementById('lbl-cust').innerText = data.customer_name;
      }

      // Play synthesized audio if available
      if (data.audio_base64) {
        const audio = new Audio("data:audio/wav;base64," + data.audio_base64);
        audio.play().catch(e => console.log("Audio autoplay prevented", e));
      }
    }

    async function toggleMic() {
      const btn = document.getElementById('mic-btn');
      if (!isRecording) {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          mediaRecorder = new MediaRecorder(stream);
          audioChunks = [];
          mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
          mediaRecorder.onstop = uploadAudio;
          mediaRecorder.start();
          isRecording = true;
          btn.classList.add('recording');
        } catch (e) {
          alert('Microphone permission denied or not available: ' + e);
        }
      } else {
        mediaRecorder.stop();
        isRecording = false;
        btn.classList.remove('recording');
      }
    }

    async function uploadAudio() {
      const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
      const reader = new FileReader();
      reader.readAsDataURL(audioBlob);
      reader.onloadend = async () => {
        const base64Audio = reader.result.split(',')[1];
        appendMsg("🎙️ [Spoken Audio Recording Sent]", 'user');
        try {
          const res = await fetch('/api/voice', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ audio: base64Audio })
          });
          const data = await res.json();
          handleAgentResponse(data);
        } catch (e) {
          appendMsg("Voice processing error.", 'agent');
        }
      };
    }
  </script>
</body>
</html>
"""

class ZentroHttpHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8")

        if self.path == "/api/chat":
            data = json.loads(body)
            msg = data.get("message", "")
            res = agent_brain.process_turn("live_web_session", msg)

            session = agent_brain.get_session("live_web_session")
            res["order_id"] = session.order_id
            if session.order_data:
                res["customer_name"] = session.order_data.get("customer_first_name", "")

            # Generate spoken speech wav
            norm = normalize_text_for_ear(res["reply"])
            out_wav = os.path.join(_HERE, "sample_outputs", "web_reply.wav")
            v_res = voice_agent.voice_turn(None, session_id="live_web_session", fallback_transcript=msg, output_audio_path=out_wav)
            if os.path.exists(out_wav):
                with open(out_wav, "rb") as f:
                    res["audio_base64"] = base64.b64encode(f.read()).decode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(res, ensure_ascii=False).encode("utf-8"))

        elif self.path == "/api/voice":
            data = json.loads(body)
            b64_audio = data.get("audio", "")
            tmp_wav = os.path.join(_HERE, "sample_outputs", "web_input.wav")
            with open(tmp_wav, "wb") as f:
                f.write(base64.b64decode(b64_audio))

            v_res = voice_agent.voice_turn(tmp_wav, session_id="live_web_session")
            session = agent_brain.get_session("live_web_session")

            resp_data = {
                "heard": v_res.get("heard", ""),
                "reply": v_res.get("reply", ""),
                "gold_evidence": v_res.get("retrieved_chunks", []),
                "verified": session.verified,
                "order_id": session.order_id
            }

            out_wav = v_res.get("output_audio_path")
            if out_wav and os.path.exists(out_wav):
                with open(out_wav, "rb") as f:
                    resp_data["audio_base64"] = base64.b64encode(f.read()).decode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(resp_data, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def start_server(port: int = 8000):
    server = HTTPServer(("0.0.0.0", port), ZentroHttpHandler)
    print("=" * 65)
    print(f"ZentroMart VoiceOps 2026: Live Web Server Running!")
    print(f"Open your browser at: http://localhost:{port}")
    print(f"Reference Date: {REFERENCE_DATE_STR} (Frozen)")
    print("Press Ctrl+C to stop.")
    print("=" * 65)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.server_close()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    start_server(port)
