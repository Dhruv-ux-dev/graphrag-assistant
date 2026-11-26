"""
Build web/demo-data.json: a snapshot of a real run for the static demo site.

Ingests the sample documents with the configured Ollama model into an in-memory
graph, answers the sample questions, and saves everything the web UI needs.

    python export_demo.py

Developed by Dhruv Mittal.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from app import DEFAULT_HOP_DEPTH, DEFAULT_MODEL, DEMO_DOCUMENTS, parse_graph_elements_via_llm, run_attributed_graph_rag
from api import SAMPLE_QUESTIONS, answer_payload
from memory_store import MemoryGraphStore


def norm(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", " ", text.lower()).strip()


def main() -> None:
    store = MemoryGraphStore()
    for name, text in DEMO_DOCUMENTS.items():
        nodes, edges = parse_graph_elements_via_llm(text, name)
        store.ingest(nodes, edges, text)
        print(f"{name}: {len(nodes)} entities, {len(edges)} relationships")

    graph = store.export_graph()
    answers, cited, valid = {}, 0, 0
    for q in SAMPLE_QUESTIONS:
        payload = answer_payload(q, run_attributed_graph_rag(q, store, DEFAULT_MODEL, DEFAULT_HOP_DEPTH))
        answers[norm(q)] = payload
        cited += len(payload["sources"]) + len(payload["invalidCitations"])
        valid += len(payload["sources"])
        print(f"\nQ: {q}\nA: {payload['answer']}\n   sources: {[s['entity'] for s in payload['sources']]}")

    documents = []
    for name, text in DEMO_DOCUMENTS.items():
        documents.append({
            "name": name, "text": text, "words": len(text.split()), "ingestedAt": None,
            "entities": sum(name in n["documents"] for n in graph["nodes"]),
            "relationships": sum(name in e["documents"] for e in graph["edges"]),
        })

    snapshot = {
        "meta": {"model": DEFAULT_MODEL, "generatedAt": datetime.now(timezone.utc).isoformat()},
        "overview": {
            "stats": {
                "documents": len(documents), "entities": len(graph["nodes"]), "relationships": len(graph["edges"]),
                "citationValidity": round(100 * valid / cited, 1) if cited else None,
            },
            "documents": documents, "graph": graph, "samples": SAMPLE_QUESTIONS, "templates": DEMO_DOCUMENTS,
        },
        "answers": answers,
    }
    out = Path(__file__).parent / "web" / "demo-data.json"
    out.write_text(json.dumps(snapshot, indent=1, ensure_ascii=False))
    print(f"\nWrote {out} ({len(graph['nodes'])} nodes, {len(graph['edges'])} edges)")


if __name__ == "__main__":
    main()
