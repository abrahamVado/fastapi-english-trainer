# English Trainer API (FastAPI)

This repository is a **learning-friendly FastAPI project** for building an English training backend.
It currently exposes endpoints for **health**, **speech‑to‑text (STT)**, **text‑to‑speech (TTS)**,
**sessions** (in‑memory), and a minimal **evaluation** scaffold. It already includes production niceties
like **config**, **logging**, and **tests**, and is structured to make new features (IPA, scoring, interview
simulator) easy to add.

> Last updated from repo scan: 2025-08-30T19:08:37.129848Z

---

## 🧭 Goals

- Clean, modular FastAPI service you can extend quickly.
- Clear separation of concerns (routers ⇄ services ⇄ schemas).
- Ready for future IPA/pronunciation scoring and interview simulation.

---

## 📁 Project structure

```
app/
├─ api/
│  ├─ deps.py                # FastAPI dependencies (DB session/auth stubs, etc.)
│  └─ routers/               # HTTP endpoints grouped by feature
│     ├─ health.py           # GET /health
│     ├─ stt.py              # POST /stt (speech → text, stubbed WhisperService)
│     ├─ tts.py              # POST /tts (text → audio, stubbed PiperService)
│     ├─ sessions.py         # CRUD-ish, in-memory sessions
│     └─ eval.py             # POST /eval (simple scoring scaffold)
├─ core/
│  ├─ config.py              # Central settings (env-driven: API_PREFIX, CORS, models, etc.)
│  ├─ logging.py             # Logging configuration
│  └─ version.py             # App name/version
├─ lifespan.py               # Lifespan hook (startup/shutdown) – currently empty
├─ main.py                   # FastAPI app; CORS; mounts routers with API_PREFIX
├─ models/
│  ├─ db.py                  # SQLAlchemy engine + session factory
│  └─ session.py             # Session ORM model (not wired yet in routers)
├─ schemas/                  # Pydantic models (requests/responses)
│  ├─ common.py              # Msg
│  ├─ eval.py                # EvalRequest, EvalResponse
│  ├─ session.py             # SessionCreate, SessionOut
│  ├─ stt.py                 # STTRequest, STTResponse
│  └─ tts.py                 # TTSRequest
├─ services/
│  ├─ eval/speech_eval.py    # EvalService (returns stub scores)
│  ├─ stt/whisper_service.py # WhisperService (stub; returns fake transcript)
│  └─ tts/piper_service.py   # PiperService (stub; yields fake wav bytes)
└─ utils/
   ├─ audio.py               # (placeholder) audio helpers
   └─ text.py                # (placeholder) text helpers

docker-compose.yml
Dockerfile
pyproject.toml
requirements.txt
uvicorn.ini
tests/
├─ conftest.py
├─ test_eval.py
├─ test_health.py
└─ test_sessions.py
```

**Separation of concerns**
- **Routers** = HTTP boundary (thin): parse inputs, call services, return schemas.
- **Services** = business logic (fat): Whisper/Piper/Eval will live here.
- **Schemas** = I/O contracts (Pydantic): strong typing and automatic docs.
- **Core** = config/logging/version; **Models** = DB entities (optional); **Utils** = helpers.

---

## ⚙️ Configuration

`app/core/config.py` defines env‑driven settings using Pydantic `BaseSettings`:

| Setting         | Default                     | Purpose |
|-----------------|-----------------------------|---------|
| `ENV`           | `dev`                       | Environment label |
| `API_PREFIX`    | `/api`                      | Prefix applied to *feature* routers (`stt`, `tts`, `eval`, `sessions`) |
| `CORS_ORIGINS`  | `["*"]`                     | Allowed origins for CORS |
| `LOG_LEVEL`     | `INFO`                      | Log level |
| `DATABASE_URL`  | `sqlite:///./english.db`    | SQLAlchemy connection string |
| `WHISPER_MODEL` | `base`                      | Whisper model name (planned) |
| `PIPER_VOICE`   | `en_US-amy-medium.onnx`     | Piper voice (planned) |

Create a `.env` in the project root to override any setting.

**CORS** is enabled in `main.py`. All feature routers are mounted at `{{API_PREFIX}}` (default `/api`).  
The **health** router is mounted without a prefix (`/health`).

---

## 🔌 API Endpoints (current)

> Shapes below are based on the code currently in the repo. STT/TTS services are stubs; replace with real implementations in `services/`.

### Health
```
GET /health
→ { "name": "english-trainer-api", "version": "0.1.0", "ok": true }
```

### STT — Speech to Text
```
POST {{API_PREFIX}}/stt
Content-Type: multipart/form-data
- audio: file (required)
- payload (optional JSON field STTRequest; unused for now)

→ 200 OK
{ "text": "<transcribed N bytes>", "duration_sec": null }
```
Implementation: `app/api/routers/stt.py` calls `WhisperService.transcribe()` (stub).  
Planned: replace with `faster-whisper`, language biasing, and confidence fields.

### TTS — Text to Speech
```
POST {{API_PREFIX}}/tts
Content-Type: application/json
{ "text": "Hello world", "voice": "en_US-amy-medium.onnx" }

→ 200 OK  (StreamingResponse)
(binary audio, media_type="audio/wav")
```
Implementation: `app/api/routers/tts.py` streams fake WAV bytes from `PiperService` (stub).  
Planned: call Piper CLI/Lib and stream chunks; support multiple voices.

### Sessions (in‑memory)
```
POST   {{API_PREFIX}}/sessions
{ "user_id": 123 }
→ { "id": 1, "user_id": 123, "status": "active" }

GET    {{API_PREFIX}}/sessions
→ [ { "id": 1, "user_id": 123, "status": "active" }, ... ]

POST   {{API_PREFIX}}/sessions/{{sid}}/end
→ { "id": 1, "user_id": 123, "status": "ended" }
```
Note: This router uses an in‑memory dict (stateless per process).  
Planned: wire to `SQLAlchemy` model `Session` and persist via `models/db.py` engine.

---

## ▶️ Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000

# Health
curl http://localhost:8000/health

# Docs
# Open in browser:
#   http://localhost:8000/docs
```

**Uvicorn options** are also available in `uvicorn.ini` (host, port, reload, workers).

---

## 🧪 Tests

```bash
pytest -q
```
- `tests/test_health.py`: basic liveness
- `tests/test_sessions.py`: sessions router
- `tests/test_eval.py`: eval scaffold

Tip: write unit tests for **services** (pure logic) and keep API tests focused.

---

## 🛣️ Roadmap / Next steps

1) **STT**: replace `WhisperService` stub with `faster-whisper` (mono 16k, VAD, biasing).  
2) **TTS**: stream Piper output; support voice selection & SSML‑like options.  
3) **Sessions**: wire to DB model (`models/session.py`) using `models/db.py` engine.  
4) **IPA**: add `/ipa/pronounce` (English→IPA→LatAm IPA + respelling).  
5) **Pronunciation scoring**: `/pron/score` (audio→phoneme accuracy vs expected text).  
6) **Interview simulator**: `/sim/*` (LLM interviewer & judge, using Ollama or GPT).  
7) **Lifespan**: warm models in `lifespan.py` (load Whisper/Piper on startup).

---

## 🧩 FastAPI concepts in this repo

- **Routers** (`APIRouter`) to group endpoints by feature.
- **Schemas** (Pydantic v2) to validate/serialize IO.
- **Dependencies** (`api/deps.py`) to inject shared resources (extend later).
- **Lifespan** (startup/shutdown) for model warm‑ups.
- **CORS & settings** centralized in `main.py` and `core/config.py`.

---

## 🔧 Replace stubs (quick guidance)

**WhisperService** (STT):  
- Use `faster-whisper` to load `{{settings.WHISPER_MODEL}}`.
- Add a small audio normalizer (mono, 16k).
- Return `{{text, duration_sec, confidence}}`.

**PiperService** (TTS):  
- Call Piper with selected voice from `{{settings.PIPER_VOICE}}`.
- Stream chunks via `StreamingResponse`.
- Return `audio/wav`.

**Sessions**:  
- Replace in‑memory `_sessions` with SQLAlchemy CRUD.
- Add `GET /sessions/{{id}}` and pagination.
