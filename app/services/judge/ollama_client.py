# app/services/judge/ollama_client.py
from __future__ import annotations

import json
import os
import re
from typing import Any, AsyncIterator, Dict, Iterator, Optional

import httpx


# ---- Env defaults -----------------------------------------------------------

DEFAULT_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_URL = os.getenv("OLLAMA_URL", "")  # if provided, can be full /api/generate
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
GENERATE_PATH = "/api/generate"
SYSTEM_TUTOR_PROMPT = (
    "You are a friendly English conversation tutor. "
    "Be clear, encouraging, and concise. Use B1–B2 vocabulary. "
    "Correct mistakes gently with 1–2 examples. "
    "Ask one short follow-up question to keep the conversation going. "
    "Avoid long monologues."
)

def _normalize_url(host: Optional[str], url: Optional[str]) -> str:
    """
    Accepts either a base host (http://host:port) or a full URL to /api/generate.
    Priority: explicit `url` > env `OLLAMA_URL` > `host`+path > DEFAULT_HOST+path.
    """
    raw_url = url or DEFAULT_URL
    if raw_url:
        # if caller already gave the full /api/generate url, keep as is
        return raw_url.rstrip("/")
    base = (host or DEFAULT_HOST).rstrip("/")
    return f"{base}{GENERATE_PATH}"


def _extract_last_json_block(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the last JSON object from a messy LLM string.
    Strategy:
      1) try fast curly-brace regex scan (balanced-ish)
      2) fallback to coarser `{ ... }` matches
    """
    # Attempt 1: scan and track brace depth
    last_obj = None
    stack = []
    start_idx = None
    for i, ch in enumerate(text):
        if ch == "{":
            stack.append(i)
            if start_idx is None:
                start_idx = i
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    chunk = text[start_idx : i + 1]
                    try:
                        last_obj = json.loads(chunk)
                    except Exception:
                        pass
                    start_idx = None
    if last_obj is not None:
        return last_obj

    # Attempt 2: coarse regex (may fail on nested braces; best-effort)
    matches = list(re.finditer(r"\{.*\}", text, flags=re.DOTALL))
    for m in reversed(matches):
        try:
            return json.loads(m.group(0))
        except Exception:
            continue
    return None


class OllamaJudge:
    """
    Minimal-yet-robust client for Ollama's /api/generate.

    Features:
      - async & sync generate methods
      - async & sync streaming methods
      - judge helpers building a JSON rubric + DATA prompt
      - env-configurable host/url/model
      - simple retries & timeouts
      - tolerant JSON extraction from LLM prose

    Environment variables (optional):
      - OLLAMA_HOST  (e.g. http://localhost:11434)
      - OLLAMA_URL   (e.g. http://localhost:11434/api/generate)  // overrides HOST
      - OLLAMA_MODEL (e.g. llama3.1)

    Example (async router):
      judge = OllamaJudge()
      result = await judge.judge_content(
          rubric='Return JSON {"score": number}.',
          data={"answer": "Hello!"}
      )

    Example (sync usage):
      judge = OllamaJudge()
      result = judge.judge_content_sync("Return JSON...", {"answer": "Hi"})
    """

    def __init__(
        self,
        host: Optional[str] = None,
        url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self.url = _normalize_url(host, url)
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout
        self.max_retries = max_retries

    # ----------------------- Low-level request helpers -----------------------

    async def _apost_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        attempt = 0
        last_exc: Optional[Exception] = None
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            while attempt <= self.max_retries:
                try:
                    r = await client.post(self.url, json=payload)
                    r.raise_for_status()
                    return r.json()
                except Exception as e:
                    last_exc = e
                    attempt += 1
            raise RuntimeError(f"Ollama request failed after {self.max_retries+1} attempts: {last_exc}")

    def _post_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        attempt = 0
        last_exc: Optional[Exception] = None
        with httpx.Client(timeout=self.timeout) as client:
            while attempt <= self.max_retries:
                try:
                    r = client.post(self.url, json=payload)
                    r.raise_for_status()
                    return r.json()
                except Exception as e:
                    last_exc = e
                    attempt += 1
            raise RuntimeError(f"Ollama request failed after {self.max_retries+1} attempts: {last_exc}")

    # ----------------------------- Generate APIs -----------------------------

    def _payload(
        self,
        prompt: str,
        model: Optional[str],
        stream: bool,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "model": model or self.model,
            "prompt": prompt,
            "stream": stream,
            # Options map to Ollama parameters. Keep them flat inside "options".
            "options": (options or {}),
        }

    # Async, non-streaming: returns response text
    async def a_generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        data = await self._apost_json(self._payload(prompt, model, stream=False, options=options))
        return data.get("response", "")

    # Sync, non-streaming
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        data = self._post_json(self._payload(prompt, model, stream=False, options=options))
        return data.get("response", "")

    # Async streaming: yields string chunks
    async def a_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        payload = self._payload(prompt, model, stream=True, options=options)
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", self.url, json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if obj.get("done"):
                        break
                    chunk = obj.get("response")
                    if chunk:
                        yield chunk

    # Sync streaming: yields string chunks
    def stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Iterator[str]:
        payload = self._payload(prompt, model, stream=True, options=options)
        with httpx.Client(timeout=None) as client:
            with client.stream("POST", self.url, json=payload) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if obj.get("done"):
                        break
                    chunk = obj.get("response")
                    if chunk:
                        yield chunk

    # ------------------------- Judge (JSON-focused) --------------------------

    def _build_judge_prompt(self, rubric: str, data: Dict[str, Any]) -> str:
        """
        Creates a prompt that strongly nudges the model to return JSON.
        You can customize the rubric to include a strict JSON schema if desired.
        """
        return (
            f"{rubric.strip()}\n\n"
            "Return ONLY a JSON object. Do not include any extra text.\n\n"
            "DATA:\n"
            f"{json.dumps(data, ensure_ascii=False, indent=2)}"
        )

    # Async judge: returns parsed dict, or a safe fallback
    async def judge_content(
        self,
        rubric: str,
        data: Dict[str, Any],
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prompt = self._build_judge_prompt(rubric, data)
        txt = await self.a_generate(prompt, model=model, options=options)
        obj = _extract_last_json_block(txt)
        return obj or {"score": 0, "key_points": [], "gaps": [], "tips": ["(ollama parse error)"]}

    # Sync judge
    def judge_content_sync(
        self,
        rubric: str,
        data: Dict[str, Any],
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prompt = self._build_judge_prompt(rubric, data)
        txt = self.generate(prompt, model=model, options=options)
        obj = _extract_last_json_block(txt)
        return obj or {"score": 0, "key_points": [], "gaps": [], "tips": ["(ollama parse error)"]}
    # ------------------------- Tutor / Chat helpers ---------------------------

    def _build_tutor_prompt(
        self, user_text: str, role: str = "", level: str = "", mode: str = ""
    ) -> str:
        context = []
        if role:  context.append(f"Role: {role}")
        if level: context.append(f"Level: {level}")
        if mode:  context.append(f"Mode: {mode}")
        ctx = "\n".join(f"- {c}" for c in context) if context else "- (none)"
        return (
            f"{SYSTEM_TUTOR_PROMPT}\n\n"
            f"Context:\n{ctx}\n\n"
            f"User said:\n{user_text}\n\n"
            f"Your response (1–3 short paragraphs max):"
        )

    async def a_tutor_reply(
        self,
        user_text: str,
        role: str = "",
        level: str = "",
        mode: str = "",
        model: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.6,
    ) -> str:
        prompt = self._build_tutor_prompt(user_text, role, level, mode)
        txt = await self.a_generate(
            prompt,
            model=model,
            options={
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        )
        return (txt or "").strip()

    def tutor_reply(
        self,
        user_text: str,
        role: str = "",
        level: str = "",
        mode: str = "",
        model: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.6,
    ) -> str:
        prompt = self._build_tutor_prompt(user_text, role, level, mode)
        txt = self.generate(
            prompt,
            model=model,
            options={
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        )
        return (txt or "").strip()
