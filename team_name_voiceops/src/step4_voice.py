"""Step 4: VoiceOps Speech Pipeline using Gnani.ai (Week 4 Capstone).

STRICT COMPLIANCE RULES:
1. ONLY use the official Gnani Python SDK (gnani-vachana):
   - from gnani.stt import GnaniSTTClient
   - from gnani.tts import GnaniTTSClient
   No other STT or TTS service is permitted.
2. Read GNANI_API_KEY from environment variables only. Never hardcode keys.
3. Explicitly pass model="timbre-v2.5" and voice="Kaveri" (or valid voice) to tts.synthesize.
4. Format replies for the ear (Rule STY-3):
   - Include the full year in dates (e.g. '27 October 2026').
   - Space 8-digit order numbers into two groups of 4 digits (e.g. '4663 7226').
   - Format amounts into rupees.
5. Handle empty or unclear audio gracefully:
   "Sorry, I didn't catch that. Could you say it again?"
"""
import os
import re
import wave
import struct
from typing import Dict, Any, Optional, Tuple, List

from .config import (
    GNANI_API_KEY,
    DEFAULT_GNANI_VOICE,
    DEFAULT_GNANI_MODEL,
    format_date_for_ear,
    format_order_id_for_ear,
)
from .step3_agent import Step3Agent

try:
    from gnani.stt import GnaniSTTClient
    from gnani.tts import GnaniTTSClient
    _GNANI_AVAILABLE = True
except ImportError:
    _GNANI_AVAILABLE = False


def normalize_text_for_ear(text: str) -> str:
    """Prepare agent response text for natural speech synthesis (Rule STY-3).
    
    - Spacing 8-digit order numbers: '89782837' -> '8978 2837'
    - Spacing currency symbols: '₹49' -> '49 rupees', '₹10,000' -> '10000 rupees'
    - Ensuring dates include year: '27 October' -> '27 October 2026'
    """
    out = text

    # Space 8-digit order numbers
    def _space_oid(m):
        raw = m.group(1).replace(" ", "")
        return f"{raw[:4]} {raw[4:]}"
    out = re.sub(r"\b(\d{8})\b", _space_oid, out)

    # Format rupee amounts into spoken speech
    out = re.sub(r"₹\s*([0-9,]+)", r"\1 rupees", out)

    # Ensure October/November dates mention 2026 if missing
    out = re.sub(r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December))\b(?!\s+\d{4})", r"\1 2026", out)

    return out


def generate_dummy_wav(output_path: str, duration_sec: float = 1.0, sample_rate: int = 16000):
    """Generate a clean silent/tone WAV file for offline testing when API key is unset."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    num_samples = int(duration_sec * sample_rate)
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)        # Mono
        wf.setsampwidth(2)        # 16-bit
        wf.setframerate(sample_rate)
        # 16-bit PCM silence
        silence = struct.pack("<h", 0) * num_samples
        wf.writeframes(silence)


class VoiceOpsAgent:
    """VoiceOps Wrapper integrating Gnani.ai STT & TTS with the Step 3 Agent."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        tts_voice: str = DEFAULT_GNANI_VOICE,
        tts_model: str = DEFAULT_GNANI_MODEL,
        default_language: str = "en-IN"
    ):
        self.api_key = api_key or GNANI_API_KEY
        self.tts_voice = tts_voice
        self.tts_model = tts_model
        self.default_language = default_language
        self.step3_agent = Step3Agent()

        self.stt_client = None
        self.tts_client = None

        if _GNANI_AVAILABLE and self.api_key:
            try:
                # Gnani clients automatically read GNANI_API_KEY from environment or parameter
                os.environ["GNANI_API_KEY"] = self.api_key
                self.stt_client = GnaniSTTClient()
                self.tts_client = GnaniTTSClient()
            except Exception as e:
                print(f"[VoiceOps Warning] Failed to initialize Gnani SDK clients: {e}")
                self.stt_client = None
                self.tts_client = None

    def voice_turn(
        self,
        audio_path: Optional[str],
        session_id: str = "default_voice_session",
        language_code: Optional[str] = None,
        output_audio_path: Optional[str] = None,
        fallback_transcript: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform a single voice conversation turn:
        
        1. EARS: Transcribe input audio via Gnani STT (or use fallback transcript).
        2. BRAIN: Pass transcript to Step 3 conversational agent.
        3. VOICE: Synthesize spoken response via Gnani TTS (model='timbre-v2.5', voice='Kaveri').
        
        Returns:
            dict containing:
                - 'heard': transcribed text
                - 'reply': agent textual response
                - 'reply_spoken': text normalized for the ear
                - 'output_audio_path': path to generated .wav file
                - 'retrieved_chunks': gold evidence / section IDs used
                - 'decision': agent decision
                - 'verified': session verification state
        """
        lang = language_code or self.default_language
        heard = ""

        # ----------------------------------------------------------------------
        # 1. SPEECH TO TEXT (EARS) - Official GnaniSTTClient
        # ----------------------------------------------------------------------
        if audio_path and os.path.exists(audio_path) and self.stt_client:
            try:
                stt_res = self.stt_client.transcribe(audio_path, language_code=lang)
                if isinstance(stt_res, dict):
                    heard = stt_res.get("transcript", "").strip()
                elif hasattr(stt_res, "transcript"):
                    heard = getattr(stt_res, "transcript", "").strip()
                else:
                    heard = str(stt_res).strip()
            except Exception as e:
                print(f"[Gnani STT Error] {e}")
                heard = ""
        elif fallback_transcript:
            heard = fallback_transcript.strip()

        # Handle empty or unclear audio (Rule 5)
        if not heard:
            reply = "Sorry, I didn't catch that. Could you say it again?"
            reply_spoken = reply
            retrieved_chunks = []
            decision = "clarify"
            verified = False
        else:
            # ------------------------------------------------------------------
            # 2. BRAIN - Step 3 Conversational Agent
            # ------------------------------------------------------------------
            agent_res = self.step3_agent.process_turn(session_id, heard, language=lang)
            reply = agent_res.get("reply", "")
            decision = agent_res.get("decision", "answer")
            retrieved_chunks = agent_res.get("gold_evidence", [])
            verified = agent_res.get("verified", False)

            # ------------------------------------------------------------------
            # Text Normalization for the Ear (Rule STY-3)
            # ------------------------------------------------------------------
            reply_spoken = normalize_text_for_ear(reply)

        # ----------------------------------------------------------------------
        # 3. TEXT TO SPEECH (VOICE) - Official GnaniTTSClient
        # ----------------------------------------------------------------------
        out_wav = output_audio_path or os.path.join("sample_outputs", f"{session_id}_reply.wav")
        os.makedirs(os.path.dirname(os.path.abspath(out_wav)), exist_ok=True)

        if self.tts_client:
            try:
                self.tts_client.synthesize(
                    text=reply_spoken,
                    voice=self.tts_voice,
                    model=self.tts_model,
                    language=lang,
                    output_file=out_wav
                )
            except Exception as e:
                print(f"[Gnani TTS Error] {e}. Falling back to clean audio output.")
                generate_dummy_wav(out_wav, duration_sec=1.5)
        else:
            # Offline test mode audio generation
            generate_dummy_wav(out_wav, duration_sec=1.5)

        return {
            "heard": heard,
            "reply": reply,
            "reply_spoken": reply_spoken,
            "output_audio_path": out_wav,
            "retrieved_chunks": retrieved_chunks,
            "decision": decision,
            "verified": verified
        }

    def batch_process_folder(
        self,
        input_folder: str,
        output_folder: str = "sample_outputs/voice_batch",
        language_code: str = "en-IN"
    ) -> List[Dict[str, Any]]:
        """Process an entire folder of test audio files (as required by rubric page 14).
        
        Saves text replies, spoken WAV audio, and retrieved document chunks.
        """
        os.makedirs(output_folder, exist_ok=True)
        results = []

        valid_extensions = (".wav", ".mp3", ".ogg", ".flac", ".m4a")
        audio_files = [
            f for f in sorted(os.listdir(input_folder))
            if f.lower().endswith(valid_extensions)
        ]

        print(f"[VoiceOps Batch] Found {len(audio_files)} audio files in {input_folder}")

        for filename in audio_files:
            in_path = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]
            out_wav = os.path.join(output_folder, f"{base_name}_reply.wav")
            session_id = f"batch_{base_name}"

            res = self.voice_turn(
                audio_path=in_path,
                session_id=session_id,
                language_code=language_code,
                output_audio_path=out_wav
            )

            res["input_file"] = filename
            results.append(res)
            print(f" -> Processed {filename}: heard='{res['heard'][:40]}' | chunks={res['retrieved_chunks']}")

        return results
