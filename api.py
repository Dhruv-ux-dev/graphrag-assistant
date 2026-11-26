"""
REST API for the GraphRAG Knowledge Assistant web UI.

Wraps the same extraction, retrieval and citation-checking code as the Streamlit
app (app.py) and serves the web UI from ./web.

Graph store: Neo4j when NEO4J_URI is set, otherwise an in-memory graph.
LLM: Ollama at OLLAMA_HOST with OLLAMA_MODEL.

Run:  uvicorn api:app --port 8000   then open http://localhost:8000

Developed by Dhruv Mittal.
"""

import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import (
    DEFAULT_HOP_DEPTH,
    DEFAULT_MODEL,
    DEMO_DOCUMENTS,
    OLLAMA_ENDPOINT,
    ExtractionError,
    Neo4jGraphStore,
    extract_search_terms,
    inference_client,
    parse_graph_elements_via_llm,
    run_attributed_graph_rag,
)
from memory_store import MemoryGraphStore

SAMPLE_QUESTIONS = [
    "Who leads the engineering team for Project Orion?",
    "Which investor backs Knowledge Graph Labs and who sits on its board?",
    "What does the Meridian Health partnership cover?",
    "Where did Priya Raman and Amit Verma work before?",
]

NEO4J_URI = os.environ.get("NEO4J_URI", "")


def _open_store():
    if NEO4J_URI:
        store = Neo4jGraphStore(NEO4J_URI, os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", ""))
        store.ensure_schema()
        return store, "neo4j"
    return MemoryGraphStore(), "memory"


store, STORE_KIND = _open_store()
lock = threading.Lock()
documents: Dict[str, Dict[str, Any]] = {}
citation_stats = {"cited": 0, "valid": 0}

app = FastAPI(title="GraphRAG Knowledge Assistant API")


class DocumentIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=20, max_length=20000)


class QuestionIn(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    depth: int = Field(default=DEFAULT_HOP_DEPTH, ge=1, le=3)


def _ollama_ok() -> bool:
    try:
        inference_client.list()
        return True
    except Exception:
        return False


def _graph() -> Dict[str, List[Dict[str, Any]]]:
    with lock:
        return store.export_graph()


def _document_list(graph: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Documents ingested in this process, plus any found only in the graph (Neo4j)."""
    counts: Dict[str, Dict[str, int]] = {}
    for n in graph["nodes"]:
        for d in n["documents"]:
            counts.setdefault(d, {"entities": 0, "relationships": 0})["entities"] += 1
    for e in graph["edges"]:
        for d in e["documents"]:
            counts.setdefault(d, {"entities": 0, "relationships": 0})["relationships"] += 1
    out = []
    for name, c in counts.items():
        meta = documents.get(name, {})
        out.append({
            "name": name,
            "words": len(meta.get("text", "").split()) or None,
            "ingestedAt": meta.get("ingestedAt"),
            "text": meta.get("text", ""),
            **c,
        })
    out.sort(key=lambda d: d.get("ingestedAt") or "", reverse=True)
    return out


def _stats(graph, docs) -> Dict[str, Any]:
    cited = citation_stats["cited"]
    return {
        "documents": len(docs),
        "entities": len(graph["nodes"]),
        "relationships": len(graph["edges"]),
        "citationValidity": round(100 * citation_stats["valid"] / cited, 1) if cited else None,
    }


@app.get("/api/status")
def status():
    return {
        "mode": "live",
        "store": STORE_KIND,
        "model": DEFAULT_MODEL,
        "ollamaEndpoint": OLLAMA_ENDPOINT,
        "ollama": _ollama_ok(),
        "graphStore": True,
    }


@app.get("/api/overview")
def overview():
    graph = _graph()
    docs = _document_list(graph)
    return {"stats": _stats(graph, docs), "documents": docs, "graph": graph,
            "samples": SAMPLE_QUESTIONS, "templates": DEMO_DOCUMENTS}


@app.post("/api/documents")
def ingest(doc: DocumentIn):
    name = " ".join(doc.name.split())
    try:
        nodes, edges = parse_graph_elements_via_llm(doc.text, name)
    except ExtractionError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    if not nodes and not edges:
        raise HTTPException(status_code=422, detail="No entities were found in this document.")
    with lock:
        store.ingest(nodes, edges, doc.text)
    documents[name] = {"text": doc.text, "ingestedAt": datetime.now(timezone.utc).isoformat()}
    return {"name": name, "entities": len(nodes), "relationships": len(edges)}


@app.delete("/api/graph")
def clear_graph():
    with lock:
        store.reset_database()
    documents.clear()
    citation_stats.update(cited=0, valid=0)
    return {"ok": True}


@app.post("/api/ask")
def ask(q: QuestionIn):
    with lock:
        result = run_attributed_graph_rag(q.question, store, DEFAULT_MODEL, q.depth)
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", result.content)})
    valid = [n for n in cited if 1 <= n <= len(result.context_entries)]
    citation_stats["cited"] += len(cited)
    citation_stats["valid"] += len(valid)
    return answer_payload(q.question, result)


def answer_payload(question: str, result) -> Dict[str, Any]:
    entries = result.context_entries
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", result.content)})
    sources = []
    for n in cited:
        if 1 <= n <= len(entries):
            e = entries[n - 1]
            sources.append({
                "n": n, "entity": e["name"], "type": e["type"], "document": e["document"],
                "evidence": e["evidence"] or e["summary"], "hops": e["hops"],
                "path": e["path"] or [e["name"]],
            })
    return {
        "question": question,
        "answer": result.content,
        "sources": sources,
        "invalidCitations": [n for n in cited if not 1 <= n <= len(entries)],
        "trace": result.execution_steps,
        "searchTerms": extract_search_terms(question),
        "contextIds": [e["id"] for e in entries],
        "citedIds": [entries[s["n"] - 1]["id"] for s in sources],
    }


WEB_DIR = Path(__file__).parent / "web"
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
