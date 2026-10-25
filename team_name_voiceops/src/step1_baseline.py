"""Step 1: Baseline Prompt-Only LLM Assistant (Weeks 1 & 2 Curriculum).

In this baseline, the assistant relies strictly on parametric memory and prompting
without access to ZentroMart's knowledge base documents or product catalog.
It demonstrates common LLM limitations (hallucinations, guessing policies, generic timelines).
"""
import os
import re
from typing import Dict, Any, Optional
from .config import (
    GEMINI_API_KEY,
    DEFAULT_GEMINI_MODEL,
)

BASELINE_SYSTEM_PROMPT = """You are ZentroMart's customer support assistant.
ZentroMart is an Indian online e-commerce store selling groceries, electronics, fashion, and books.
Rules:
1. Introduce yourself as ZentroMart's AI support assistant in your first reply.
2. Reply politely in 2-3 short sentences.
3. Match the language used by the customer (English, Hindi, or Hinglish).
4. Do your best to help the customer with returns, refunds, delivery, and warranties.
"""

# Deterministic simulated parametric baseline responses for offline testing & benchmark consistency
# (reflects an LLM with no knowledge of ZentroMart's internal documents)
_PARAMETRIC_BASELINES = {
    "refund_wallet": "Refunds usually take 5 to 7 business days to reflect in your ZentroMart Wallet.",
    "return_window_general": "Generally, items can be returned within 30 days of delivery through our app.",
    "student_discount": "Yes, we offer a 10% student discount if you verify your student ID on our portal.",
    "midnight_delivery": "Yes, we provide 24/7 midnight delivery across all metro cities for a small surcharge.",
    "gst_bulk": "Yes, for bulk orders you can enter your company GSTIN at checkout to receive a B2B tax invoice.",
    "default": "Hello! I am ZentroMart's AI support assistant. I would be happy to help you with your inquiry. Please let me know how I can assist you further."
}

class Step1BaselineAgent:
    """Prompt-only baseline agent without RAG retrieval."""

    def __init__(self, model_name: str = DEFAULT_GEMINI_MODEL, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or GEMINI_API_KEY
        self.client = None

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception:
                try:
                    import google.generativeai as gai
                    gai.configure(api_key=self.api_key)
                    self.client = gai.GenerativeModel(self.model_name)
                except Exception:
                    self.client = None

    def respond(self, query: str, language: str = "en") -> Dict[str, Any]:
        """Generate a response using prompt-only LLM.
        
        Returns:
            dict containing:
                - 'decision': 'answer' or 'clarify' or 'escalate'
                - 'reply': response string
                - 'retrieved_chunks': [] (empty for Step 1)
        """
        prompt = f"Customer Query: {query}\nProvide a helpful customer support reply."

        if self.client:
            try:
                # Try official google.genai client
                if hasattr(self.client, "models"):
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config={"system_instruction": BASELINE_SYSTEM_PROMPT, "temperature": 0.0}
                    )
                    text = response.text.strip()
                    return {
                        "decision": "answer",
                        "reply": text,
                        "retrieved_chunks": [],
                        "model": self.model_name,
                        "grounded": False
                    }
                elif hasattr(self.client, "generate_content"):
                    # Fallback to google.generativeai
                    response = self.client.generate_content(
                        prompt,
                        generation_config={"temperature": 0.0}
                    )
                    text = response.text.strip()
                    return {
                        "decision": "answer",
                        "reply": text,
                        "retrieved_chunks": [],
                        "model": self.model_name,
                        "grounded": False
                    }
            except Exception:
                pass

        # Offline / Parametric Fallback Mode
        q_lower = query.lower()
        if "wallet" in q_lower and "refund" in q_lower:
            reply = "Hello! I am ZentroMart's AI assistant. " + _PARAMETRIC_BASELINES["refund_wallet"]
        elif "student discount" in q_lower:
            reply = "Hello! I am ZentroMart's AI assistant. " + _PARAMETRIC_BASELINES["student_discount"]
        elif "midnight" in q_lower:
            reply = "Hello! I am ZentroMart's AI assistant. " + _PARAMETRIC_BASELINES["midnight_delivery"]
        elif "gst" in q_lower:
            reply = "Hello! I am ZentroMart's AI assistant. " + _PARAMETRIC_BASELINES["gst_bulk"]
        elif "return" in q_lower or "lauta" in q_lower or "wapas" in q_lower:
            reply = "Hello! I am ZentroMart's AI assistant. " + _PARAMETRIC_BASELINES["return_window_general"]
        else:
            reply = _PARAMETRIC_BASELINES["default"]

        return {
            "decision": "answer",
            "reply": reply,
            "retrieved_chunks": [],
            "model": "parametric_baseline_offline",
            "grounded": False
        }
