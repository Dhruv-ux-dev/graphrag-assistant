"""
In-memory knowledge graph with the same interface as Neo4jGraphStore.

Lets the assistant run without a Neo4j server (local demos, tests, the static
snapshot export). Matching and traversal follow the Cypher queries in app.py:
entry nodes are ranked by name, summary and source-text matches, and traversal
returns each neighbour once with its shortest path.

Developed by Dhruv Mittal.
"""

from collections import deque
from typing import Any, Dict, List, Tuple

from app import (
    DEFAULT_HOP_DEPTH,
    MAX_ENTRY_NODES,
    GraphEdge,
    GraphNode,
    extract_search_terms,
    find_snippet,
    make_node_id,
    normalize_text,
)


class MemoryGraphStore:
    """Dictionary-backed graph store; safe for a single process."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}
        # (source_id, target_id, type) -> edge record
        self.edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    def __enter__(self) -> "MemoryGraphStore":
        return self

    def __exit__(self, *exc_info) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def ensure_schema(self) -> None:
        pass

    def reset_database(self) -> None:
        self.nodes.clear()
        self.edges.clear()

    def graph_size(self) -> Tuple[int, int]:
        return len(self.nodes), len(self.edges)

    # ---- Writes -------------------------------------------------------------

    def _merge_node(self, node_id: str, name: str, category: str, summary: str,
                    document: str, snippet: str) -> None:
        node = self.nodes.get(node_id)
        if node is None:
            self.nodes[node_id] = node = {
                "id": node_id, "name": name, "key": normalize_text(name),
                "category": category, "summary": summary,
                "documents": [], "snippets": [],
            }
        else:
            if not node["summary"]:
                node["summary"] = summary
            if node["category"] == "CONCEPT":
                node["category"] = category
        if document not in node["documents"]:
            node["documents"].append(document)
            node["snippets"].append(snippet)

    def ingest(self, nodes: List[GraphNode], edges: List[GraphEdge], document_content: str) -> None:
        for n in nodes:
            self._merge_node(n.node_id, n.display_name, n.category, n.summary,
                             n.origin_document, n.text_snippet)
        for e in edges:
            sid, tid = make_node_id(e.source_node), make_node_id(e.target_node)
            # Endpoints the model forgot to list as nodes are still created
            if sid not in self.nodes:
                self._merge_node(sid, e.source_node, "CONCEPT", "", e.origin_document,
                                 find_snippet(document_content, e.source_node))
            if tid not in self.nodes:
                self._merge_node(tid, e.target_node, "CONCEPT", "", e.origin_document,
                                 find_snippet(document_content, e.target_node))
            key = (sid, tid, e.edge_type)
            edge = self.edges.get(key)
            if edge is None:
                self.edges[key] = {"source": sid, "target": tid, "type": e.edge_type,
                                   "context": e.context, "documents": [e.origin_document]}
            elif e.origin_document not in edge["documents"]:
                edge["documents"].append(e.origin_document)

    # ---- Reads --------------------------------------------------------------

    def find_entry_nodes(self, question: str, limit: int = MAX_ENTRY_NODES) -> List[Dict]:
        terms = [f" {t}" for t in extract_search_terms(question)]
        padded_question = f" {normalize_text(question)} "
        matches = []
        for n in self.nodes.values():
            key = f" {n['key']} "
            summary = f" {normalize_text(n['summary'])} "
            evidence = f" {normalize_text(' '.join(n['snippets']))} "
            named = bool(n["key"]) and key in padded_question
            name_hits = sum(t in key for t in terms)
            summary_hits = sum(t in summary for t in terms)
            evidence_hits = sum(t in evidence for t in terms)
            if not (named or name_hits or summary_hits or evidence_hits):
                continue
            matches.append({
                "id": n["id"], "name": n["name"], "summary": n["summary"],
                "type": n["category"], "documents": list(n["documents"]),
                "snippets": list(n["snippets"]),
                "name_match": named or name_hits > 0,
                "relevance": (10 if named else 0) + name_hits * 3 + summary_hits * 2 + evidence_hits,
            })
        matches.sort(key=lambda m: (not m["name_match"], -m["relevance"], -len(m["name"])))
        matches = matches[: limit * 3]
        if any(m["name_match"] for m in matches):
            matches = [m for m in matches if m["name_match"]]
        return matches[:limit]

    def query_nodes_by_text(self, text_query: str) -> List[Dict]:
        return self.find_entry_nodes(text_query, limit=12)

    def fetch_hops_context(self, node_id: str, depth: int = DEFAULT_HOP_DEPTH, limit: int = 25) -> List[Dict]:
        """Breadth-first search over edges in both directions, shortest path per neighbour."""
        if node_id not in self.nodes:
            return []
        depth = max(1, min(int(depth), 5))
        adjacency: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        for edge in self.edges.values():
            adjacency.setdefault(edge["source"], []).append((edge["target"], edge))
            adjacency.setdefault(edge["target"], []).append((edge["source"], edge))

        def describe(edge: Dict[str, Any]) -> str:
            text = f"{self.nodes[edge['source']]['name']} -[{edge['type']}]-> {self.nodes[edge['target']]['name']}"
            return text + (f": {edge['context']}" if edge["context"] else "")

        seen = {node_id: ([node_id], [])}
        queue = deque([node_id])
        while queue:
            current = queue.popleft()
            path, rels = seen[current]
            if len(path) - 1 >= depth:
                continue
            for neighbour, edge in sorted(adjacency.get(current, []), key=lambda x: self.nodes[x[0]]["name"].lower()):
                if neighbour in seen:
                    continue
                seen[neighbour] = (path + [neighbour], rels + [describe(edge)])
                queue.append(neighbour)

        results = []
        for nid, (path, rels) in seen.items():
            if nid == node_id:
                continue
            n = self.nodes[nid]
            results.append({
                "id": nid, "name": n["name"], "summary": n["summary"], "type": n["category"],
                "documents": list(n["documents"]), "snippets": list(n["snippets"]),
                "hops": len(path) - 1,
                "path_names": [self.nodes[p]["name"] for p in path],
                "relations": rels,
            })
        results.sort(key=lambda r: (r["hops"], r["name"]))
        return results[:limit]

    def load_graph(self, graph: Dict[str, List[Dict[str, Any]]], texts: Dict[str, str]) -> None:
        """Load nodes and edges exported by export_graph(); snippets are rebuilt from the document texts."""
        for n in graph["nodes"]:
            self.nodes[n["id"]] = {
                "id": n["id"], "name": n["name"], "key": normalize_text(n["name"]),
                "category": n["type"], "summary": n.get("summary", ""),
                "documents": list(n["documents"]),
                "snippets": [find_snippet(texts.get(d, ""), n["name"]) for d in n["documents"]],
            }
        for e in graph["edges"]:
            self.edges[(e["source"], e["target"], e["type"])] = {
                "source": e["source"], "target": e["target"], "type": e["type"],
                "context": e.get("context", ""), "documents": list(e["documents"]),
            }

    def export_graph(self) -> Dict[str, List[Dict[str, Any]]]:
        """All nodes and edges, for the graph view."""
        return {
            "nodes": [
                {"id": n["id"], "name": n["name"], "type": n["category"], "summary": n["summary"],
                 "documents": list(n["documents"])}
                for n in self.nodes.values()
            ],
            "edges": [
                {"source": e["source"], "target": e["target"], "type": e["type"],
                 "context": e["context"], "documents": list(e["documents"])}
                for e in self.edges.values()
            ],
        }
