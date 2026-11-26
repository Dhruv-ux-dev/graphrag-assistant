"""
Hosted LLM client for OpenAI-compatible APIs (Google Gemini, Groq).

Exposes the same small surface the app uses from the Ollama client:
`chat(model, messages, format=None, options=None, **_)` returning
{"message": {"content": ...}}, and `list()`.

Developed by Dhruv Mittal.
"""

import os
import re
import time
from typing import Any, Dict, List, Optional

import httpx

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


RETRYABLE = {429, 500, 502, 503, 504}


class OpenAICompatClient:
    def __init__(self, api_key: str, base_url: str, label: str, timeout: float = 45.0) -> None:
        self.label = label
        # Models tried, in order, when the requested one is overloaded or unavailable
        self.fallback_models: List[str] = []
        self.last_model = ""
        self._http = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def chat(self, model: str, messages: List[Dict[str, str]], format: Optional[str] = None,
             options: Optional[Dict[str, Any]] = None, **_: Any) -> Dict[str, Any]:
        last_error = ""
        for candidate in [model] + [m for m in self.fallback_models if m != model]:
            for attempt in range(1):
                body: Dict[str, Any] = {"model": candidate, "messages": messages}
                if options and "temperature" in options:
                    body["temperature"] = options["temperature"]
                if format == "json":
                    body["response_format"] = {"type": "json_object"}
                try:
                    res = self._http.post("chat/completions", json=body)
                except httpx.TimeoutException:
                    last_error = f"{candidate} timed out"
                    break  # try the next model
                if res.status_code < 400:
                    self.last_model = candidate
                    return {"message": {"content": res.json()["choices"][0]["message"]["content"]}}
                last_error = f"{self.label} API error {res.status_code} ({candidate}): {res.text[:200]}"
                if res.status_code == 404:
                    break
                if res.status_code not in RETRYABLE:
                    raise RuntimeError(last_error)
                time.sleep(1)
        raise RuntimeError(last_error or f"{self.label} API request failed")

    def list(self) -> Dict[str, Any]:
        res = self._http.get("models")
        res.raise_for_status()
        return {"models": [{"model": m["id"]} for m in res.json().get("data", [])]}


def flash_models(client: "OpenAICompatClient") -> List[str]:
    """Stable Gemini Flash models the key can use: newest 'flash' first, then 'flash-lite'."""
    try:
        ids = [m["model"].split("/")[-1] for m in client.list()["models"]]
    except Exception:
        return []
    ranked = []
    for i in ids:
        m = re.fullmatch(r"gemini-(\d+(?:\.\d+)?)-flash(-lite)?", i)
        if m:
            version = tuple(int(x) for x in m.group(1).split("."))
            ranked.append((m.group(2) is None, version, i))
    ranked.sort(reverse=True)
    return [i for _, _, i in ranked]


def pick_latest_flash(client: "OpenAICompatClient", fallback: str) -> str:
    """Newest stable 'gemini-X.Y-flash' model the key can use, or `fallback` if the list is unavailable."""
    models = flash_models(client)
    full = [m for m in models if not m.endswith("-lite")]
    return full[0] if full else (models[0] if models else fallback)
