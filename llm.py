"""
Hosted LLM client for OpenAI-compatible APIs (Google Gemini, Groq).

Exposes the same small surface the app uses from the Ollama client:
`chat(model, messages, format=None, options=None, **_)` returning
{"message": {"content": ...}}, and `list()`.

Developed by Dhruv Mittal.
"""

import os
from typing import Any, Dict, List, Optional

import httpx

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class OpenAICompatClient:
    def __init__(self, api_key: str, base_url: str, label: str, timeout: float = 90.0) -> None:
        self.label = label
        self._http = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def chat(self, model: str, messages: List[Dict[str, str]], format: Optional[str] = None,
             options: Optional[Dict[str, Any]] = None, **_: Any) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": model, "messages": messages}
        if options and "temperature" in options:
            body["temperature"] = options["temperature"]
        if format == "json":
            body["response_format"] = {"type": "json_object"}
        res = self._http.post("chat/completions", json=body)
        if res.status_code >= 400:
            raise RuntimeError(f"{self.label} API error {res.status_code}: {res.text[:300]}")
        return {"message": {"content": res.json()["choices"][0]["message"]["content"]}}

    def list(self) -> Dict[str, Any]:
        res = self._http.get("models")
        res.raise_for_status()
        return {"models": [{"model": m["id"]} for m in res.json().get("data", [])]}
