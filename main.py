"""
ContextIQ CLI, Benchmark Evaluation, and REST API Entrypoint.

Commands:
    python main.py                              # Interactive question REPL for admins
    python main.py --question "How do I reset password?"
    python main.py --evaluate                   # Automated benchmark against example_questions.json
    python main.py --ingest <json_file>         # Ingest a new document or update corpus
    python main.py --serve                      # Launch FastAPI REST API on Uvicorn
    python main.py --serve --port 9000          # Custom port
    python main.py --serve --host 0.0.0.0       # Bind to all interfaces (Docker/ECS)
"""
import argparse
import json
import os
import sys
from typing import Dict, Any

from contextiq.engine import ContextIQEngine
from contextiq.models import Document

DEFAULT_DOCS_PATH = os.path.join("starter_kit", "data", "docs.json")
DEFAULT_QUESTIONS_PATH = os.path.join("starter_kit", "data", "example_questions.json")


def format_response_cli(resp) -> str:
    """Formats response payload cleanly for console output."""
    lines = []
    lines.append("=" * 72)
    lines.append(f"QUESTION: {resp.question}")
    lines.append("-" * 72)
    if resp.status == "ANSWERED" and resp.citation:
        lines.append(f"STATUS:        [OK] ANSWERED")
        lines.append(f"CONFIDENCE:    {resp.confidence_tier} (Score: {resp.confidence_score:.3f} >= Threshold: {resp.effective_threshold:.3f})")
        lines.append(f"SOURCE DOC:    [{resp.citation.doc_id}] {resp.citation.title}")
        if resp.citation.navigation_path:
            lines.append(f"UI NAVIGATION: {resp.citation.navigation_path}")
        lines.append(f"DIRECT ANSWER:\n  \"{resp.answer}\"")
    else:
        lines.append(f"STATUS:        [REFUSAL] I DON'T KNOW")
        lines.append(f"CONFIDENCE:    {resp.confidence_tier} (Score: {resp.confidence_score:.3f} < Threshold: {resp.effective_threshold:.3f})")
        if resp.rejection_reason:
            lines.append(f"REASON:        {resp.rejection_reason}")
        lines.append(f"RESPONSE:\n  {resp.answer}")
        if resp.escalation_path:
            lines.append(f"ESCALATION:    {resp.escalation_path}")
    lines.append("=" * 72)
    return "\n".join(lines)


def run_evaluation(engine: ContextIQEngine, questions_path: str) -> None:
    """Runs automated evaluation against the benchmark dataset."""
    print("\n" + "=" * 80)
    print("                CONTEXTIQ BENCHMARK EVALUATION RUNNER")
    print("=" * 80)

    if not os.path.exists(questions_path):
        print(f"Error: Evaluation file not found at '{questions_path}'")
        return

    with open(questions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("example_questions", [])
    total = len(questions)
    passed = 0
    hallucinations = 0

    print(f"Loaded {total} benchmark questions from: {questions_path}\n")

    for idx, item in enumerate(questions, 1):
        q_id = item.get("question_id", f"Q-{idx}")
        q_text = item.get("question", "")
        expected_doc_id = item.get("expected_doc_id")
        note = item.get("note", "")

        resp = engine.ask(q_text)

        # Verification:
        # If expected_doc_id is None: must REFUSE to answer (any answer is a hallucination!)
        # If expected_doc_id is provided: must ANSWER with that exact doc_id
        is_correct = False
        if expected_doc_id is None:
            if resp.status == "REFUSED":
                is_correct = True
            else:
                is_correct = False
                hallucinations += 1
        else:
            is_correct = (resp.status == "ANSWERED" and resp.citation is not None and resp.citation.doc_id == expected_doc_id)

        if is_correct:
            passed += 1
            verdict = "[PASS]"
        else:
            verdict = "[FAIL]"

        actual_doc = resp.citation.doc_id if resp.citation else None
        print(f"{verdict} {q_id}: \"{q_text}\"")
        print(f"       Expected Doc: {str(expected_doc_id):<6} | Actual Doc: {str(actual_doc):<6} | Status: {resp.status}")
        print(f"       Score: {resp.confidence_score:.3f} | Threshold: {resp.effective_threshold:.3f} | Tier: {resp.confidence_tier}")
        if resp.status == "ANSWERED":
            print(f"       Direct Answer: \"{resp.answer}\"")
        else:
            print(f"       Safe Refusal:  \"{resp.answer.splitlines()[0]}\"")
            if resp.rejection_reason:
                print(f"       Rejection Log: {resp.rejection_reason}")
        if note:
            print(f"       Benchmark Note: {note}")
        print("-" * 80)

    accuracy = (passed / total) * 100 if total > 0 else 0
    hallucination_rate = (hallucinations / total) * 100 if total > 0 else 0

    print("\n" + "=" * 80)
    print("                       BENCHMARK SCORECARD")
    print("=" * 80)
    print(f"  Total Evaluated:       {total}")
    print(f"  Passed Verifications:  {passed} / {total}")
    print(f"  Grounding Accuracy:    {accuracy:.1f}%")
    print(f"  Hallucination Rate:    {hallucination_rate:.1f}%")
    print("=" * 80 + "\n")


def interactive_repl(engine: ContextIQEngine) -> None:
    """Starts interactive terminal session for corporate administrators."""
    stats = engine.get_stats()
    print("\n" + "=" * 72)
    print(" Welcome to ContextIQ - Solstice Learning Admin Assistant")
    print(f" Status: Active ({stats['document_count']} documents, {stats['chunk_count']} chunks indexed)")
    print(" Commands:")
    print("   Type your question in plain English")
    print("   '/stats'   - View index statistics")
    print("   '/reload'  - Reload documentation corpus from disk")
    print("   '/quit'    - Exit the assistant")
    print("=" * 72 + "\n")

    while True:
        try:
            q = input("\nAdmin Question > ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit", "q", "/quit"}:
                print("Exiting ContextIQ. Have a great day!")
                break
            elif q.lower() == "/stats":
                curr = engine.get_stats()
                print(f"[Stats] Docs: {curr['document_count']}, Chunks: {curr['chunk_count']}, Vocab: {curr['vocabulary_size']} words")
                continue
            elif q.lower() == "/reload":
                reloaded = engine.load_docs(DEFAULT_DOCS_PATH)
                print(f"[System] Successfully reloaded {reloaded} chunks from {DEFAULT_DOCS_PATH}!")
                continue

            resp = engine.ask(q)
            print(format_response_cli(resp))
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


def main():
    parser = argparse.ArgumentParser(description="ContextIQ: Documentation Q&A Assistant")
    parser.add_argument("--docs", default=DEFAULT_DOCS_PATH, help="Path to docs.json")
    parser.add_argument("--question", "-q", type=str, help="Single question to answer")
    parser.add_argument("--evaluate", "-e", action="store_true", help="Run benchmark against example_questions.json")
    parser.add_argument("--questions-file", default=DEFAULT_QUESTIONS_PATH, help="Path to evaluation questions file")
    parser.add_argument("--ingest", type=str, help="Path to a single document JSON to ingest into the corpus")
    parser.add_argument("--serve", action="store_true", help="Launch FastAPI REST API server via Uvicorn")
    parser.add_argument("--host", default="127.0.0.1", help="API server bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="API server port (default: 8000)")
    args = parser.parse_args()

    # --serve mode: launch Uvicorn and exit (does not initialize CLI engine)
    if args.serve:
        try:
            import uvicorn
        except ImportError:
            print("Error: uvicorn is required for --serve mode.")
            print("Install with: pip install uvicorn[standard]")
            sys.exit(1)
        print(f"Starting ContextIQ API server on {args.host}:{args.port}")
        print(f"Swagger UI: http://{args.host}:{args.port}/docs")
        uvicorn.run("api:app", host=args.host, port=args.port, reload=False)
        return

    engine = ContextIQEngine(args.docs)

    if args.ingest:
        with open(args.ingest, "r", encoding="utf-8") as f:
            doc_data = json.load(f)
        doc = Document(
            doc_id=doc_data.get("doc_id", "CUSTOM"),
            title=doc_data.get("title", "Untitled"),
            content=doc_data.get("content", ""),
            metadata=doc_data.get("metadata", {}),
        )
        total_chunks = engine.ingest_document(doc)
        print(f"Document [{doc.doc_id}] '{doc.title}' ingested successfully. Total chunks: {total_chunks}")
    elif args.evaluate:
        run_evaluation(engine, args.questions_file)
    elif args.question:
        resp = engine.ask(args.question)
        print(format_response_cli(resp))
    else:
        interactive_repl(engine)


if __name__ == "__main__":
    main()
