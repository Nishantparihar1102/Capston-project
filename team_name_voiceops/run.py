#!/usr/bin/env python3
"""Unified CLI Entry Point for VoiceOps 2026 (ZentroMart Customer Support Agent).

Supports:
1. Batch evaluation of audio files:
   python run.py --input-dir <audio_folder> --output-dir sample_outputs/
2. Interactive text / voice session:
   python run.py --interactive
3. Single query execution:
   python run.py --query "How fast are refunds to the ZentroMart Wallet?"
4. Automated benchmark evaluation:
   python run.py --eval
5. Specific step demonstration:
   python run.py --step [1|2|3|4|all]
"""
import os
import sys
import argparse
import json

# Ensure team_name_voiceops is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from src.config import REFERENCE_DATE_STR, SAMPLE_OUTPUTS_DIR
from src.step1_baseline import Step1BaselineAgent
from src.step2_rag import Step2RAGAgent
from src.step3_agent import Step3Agent
from src.step4_voice import VoiceOpsAgent
from src.evaluate import run_all_evaluations


def run_interactive():
    """Start an interactive multi-turn terminal conversation with Step 3 / Step 4 agent."""
    print("=" * 65)
    print("ZentroMart VoiceOps 2026 - Interactive Support Agent")
    print(f"Reference Date: {REFERENCE_DATE_STR} (Frozen for Capstone)")
    print("Type 'exit' or 'quit' to end the session.")
    print("=" * 65)

    agent = Step3Agent()
    session_id = "interactive_cli_session"

    # Turn 1 Greeting
    greeting = agent.process_turn(session_id, "Hello")
    print(f"\n[Agent]: {greeting['reply']}\n")

    while True:
        try:
            user_input = input("[Customer]: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\n[Agent]: Thank you for contacting ZentroMart. Have a great day!\n")
                break

            res = agent.process_turn(session_id, user_input)
            print(f"\n[Agent]: {res['reply']}")
            if res.get("tool_calls"):
                print(f" -> Tools Executed: {res['tool_calls']}")
            if res.get("gold_evidence"):
                print(f" -> Policy Evidence: {res['gold_evidence']}")
            print()
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


def main():
    parser = argparse.ArgumentParser(
        description="VoiceOps 2026: ZentroMart AI Customer Support Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Run batch voice evaluation on audio files:
  python run.py --input-dir ./test_audio/ --output-dir ./sample_outputs/

  # Run full automated benchmark:
  python run.py --eval

  # Ask a single question:
  python run.py --query "How fast are refunds to ZentroMart Wallet?"

  # Launch interactive chat:
  python run.py --interactive
"""
    )
    parser.add_argument("--input-dir", type=str, help="Folder containing customer test audio files (.wav, .mp3, etc.)")
    parser.add_argument("--output-dir", type=str, default=SAMPLE_OUTPUTS_DIR, help="Destination folder for replies and logs")
    parser.add_argument("--query", type=str, help="Single test customer query")
    parser.add_argument("--step", type=str, choices=["1", "2", "3", "4", "all"], help="Execute specific milestone step demo")
    parser.add_argument("--interactive", action="store_true", help="Start an interactive multi-turn CLI session")
    parser.add_argument("--eval", action="store_true", help="Run the complete automated benchmark suite on practice datasets")
    parser.add_argument("--language", type=str, default="en-IN", help="Language code (e.g. en-IN, hi-IN, hi-en)")

    args = parser.parse_args()

    # 1. Batch Audio Evaluation Mode (Primary Rubric Run Command)
    if args.input_dir:
        print(f"\n[VoiceOps] Running batch audio processing on: {args.input_dir}")
        print(f"[VoiceOps] Saving outputs to: {args.output_dir}\n")
        voice_agent = VoiceOpsAgent(default_language=args.language)
        results = voice_agent.batch_process_folder(
            input_folder=args.input_dir,
            output_folder=args.output_dir,
            language_code=args.language
        )
        summary_file = os.path.join(args.output_dir, "batch_audio_results.json")
        with open(summary_file, mode="w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[VoiceOps] Batch processing completed. Summary saved to {summary_file}")
        return

    # 2. Automated Benchmark Mode
    if args.eval:
        run_all_evaluations(args.output_dir)
        return

    # 3. Interactive Session
    if args.interactive:
        run_interactive()
        return

    # 4. Single Query Execution
    if args.query:
        print(f"\nQuery: {args.query}")
        step2_agent = Step2RAGAgent()
        res = step2_agent.respond(args.query, language=args.language)
        print(f"Decision: {res['decision']}")
        print(f"Retrieved Sections: {res['retrieved_chunks']}")
        print(f"Reply: {res['reply']}\n")
        return

    # 5. Milestone Step Demo
    if args.step:
        if args.step == "1":
            print("\n--- STEP 1: PROMPT-ONLY BASELINE DEMO ---")
            s1 = Step1BaselineAgent()
            res = s1.respond("How fast are refunds to the ZentroMart Wallet?")
            print(f"Query: How fast are refunds to the ZentroMart Wallet?")
            print(f"Reply: {res['reply']}\n(Note: Guesses 5-7 days due to lack of document grounding)")
        elif args.step == "2":
            print("\n--- STEP 2: GROUNDED RAG DEMO ---")
            s2 = Step2RAGAgent()
            res = s2.respond("How fast are refunds to the ZentroMart Wallet?")
            print(f"Query: How fast are refunds to the ZentroMart Wallet?")
            print(f"Retrieved Evidence: {res['retrieved_chunks']}")
            print(f"Reply: {res['reply']}\n(Note: Grounded in REF-2, avoids FAQ-1 trap)")
        elif args.step == "3":
            print("\n--- STEP 3: MULTI-TURN CONVERSATION & IDENTITY GATE DEMO ---")
            s3 = Step3Agent()
            sid = "demo_step3"
            t1 = s3.process_turn(sid, "Hi, I need help with order 89782837")
            print("Turn 1:", t1["reply"])
            t2 = s3.process_turn(sid, "Last four digits are 4533")
            print("Turn 2:", t2["reply"])
            t3 = s3.process_turn(sid, "Actually, can I speak to a human?")
            print("Turn 3:", t3["reply"], "| Tools:", t3["tool_calls"])
        elif args.step == "4":
            print("\n--- STEP 4: VOICEOPS SPEECH PIPELINE DEMO ---")
            v4 = VoiceOpsAgent()
            res = v4.voice_turn(None, session_id="step4_demo", fallback_transcript="My express order arrived late. Do I get anything back?")
            print("Transcript Heard:", res["heard"])
            print("Spoken Output Normalized:", res["reply_spoken"])
            print("Audio Output Generated:", res["output_audio_path"])
        elif args.step == "all":
            run_all_evaluations(args.output_dir)
        return

    # Default if no arguments: print help and run evaluation
    parser.print_help()
    print("\nRunning default evaluation suite now:\n")
    run_all_evaluations(args.output_dir)


if __name__ == "__main__":
    main()
