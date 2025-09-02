# app/utils/idempotency.py
from __future__ import annotations
import time, hashlib, io
from typing import Any, Optional, Tuple
from cachetools import TTLCache

# --- Simple TTL caches (in-memory). For multi-process, swap to Redis. ---

RESP_TTL_SECONDS = 600
RESP_MAX = 2048
_response_cache = TTLCache(maxsize=RESP_MAX, ttl=RESP_TTL_SECONDS)

SEEN_TTL_SECONDS = 600
SEEN_MAX = 8192
_seen_ids = TTLCache(maxsize=SEEN_MAX, ttl=SEEN_TTL_SECONDS)

AUDIO_TTL_SECONDS = 900
AUDIO_MAX = 4096
_audio_fingerprints = TTLCache(maxsize=AUDIO_MAX, ttl=AUDIO_TTL_SECONDS)

def _sha256(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def idempotency_hit(req_id: Optional[str]) -> Tuple[bool, Optional[Any], str]:
    """
    Returns (hit, cached_value, reason). If cached_value is not None, reuse it.
    """
    if not req_id:
        return False, None, "missing-req-id"
    if req_id in _response_cache:
        return True, _response_cache[req_id], "response-cache"
    if req_id in _seen_ids:
        return True, None, "seen-in-flight"
    return False, None, "miss"

def idempotency_mark_seen(req_id: Optional[str]) -> None:
    if req_id:
        _seen_ids[req_id] = time.time()

def idempotency_store(req_id: Optional[str], value: Any) -> None:
    if req_id:
        _response_cache[req_id] = value
        _seen_ids[req_id] = time.time()

def is_duplicate_audio(session_id: str, question_id: str, audio_bytes: bytes) -> bool:
    """
    Returns True if the exact same audio content for the same (session, question)
    was seen recently.
    """
    if not session_id or not question_id:
        return False
    key = f"{session_id}:{question_id}"
    fp = _sha256(audio_bytes)
    prev = _audio_fingerprints.get(key)
    if prev == fp:
        return True
    _audio_fingerprints[key] = fp
    return False

# --- Whisper segment de-repeat helper --------------------------------------

def stitch_unique(segments) -> str:
    """
    Given faster-whisper/OpenAI-Whisper-like segments with .text, produce
    text with simple repetition suppression (exact + short overlaps).
    """
    out = []
    last = ""
    for s in segments:
        t = (getattr(s, "text", None) or str(s)).strip()
        if not t:
            continue
        if t == last:
            continue
        if last and (t.endswith(last) or last.endswith(t)):
            longer = t if len(t) >= len(last) else last
            if out:
                out[-1] = longer
            else:
                out.append(longer)
            last = longer
            continue
        out.append(t)
        last = t
    return " ".join(out)
