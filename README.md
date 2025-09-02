# English Trainer API (FastAPI)

# English Trainer API (FastAPI)

This repository contains a **full-stack English training platform** built with **FastAPI** (backend) and **React/Vite** (frontend).  
It provides GPU-accelerated **speech-to-text (STT)** via *faster-whisper*, high-quality **text-to-speech (TTS)** with *Bark*, and experimental modules for **IPA conversion**, **pronunciation scoring**, and **interview simulation**.  

The backend is designed with **clean separation of concerns** (routers ⇄ services ⇄ schemas), production-ready features like **config**, **logging**, **Docker**, and **tests**, while the frontend offers an interactive web app for practicing conversations and playback.

Endpoints currently include **health checks**, **STT**, **TTS**, **sessions** (in-memory), **evaluation**, and **simulation** (LLM-based interviewer/judge in progress).


> Last updated from repo scan: 2025-09-02T00:00:00Z

---

## 🧭 Goals

- Provide a **full-stack English training platform** (FastAPI backend + React/Vite frontend).  
- Clean, modular FastAPI service with a **clear separation of concerns** (routers ⇄ services ⇄ schemas).  
- Leverage **GPU acceleration** for real-time Whisper STT and Bark TTS.  
- Support **IPA conversion**, **pronunciation scoring**, and **interview simulation** as core features.  
- Include production-ready tools: **config, logging, Docker, tests**, and **developer-friendly docs**.  
- Offer an **interactive web app** to practice conversations, listen to synthesized speech, and review results.  


---

## 📁 Project structure

```
app/                          # FastAPI backend
├─ api/                       # HTTP endpoints
│  ├─ deps.py                 # FastAPI dependencies (DB/auth stubs)
│  └─ routers/                # Feature routers
│     ├─ health.py            # GET /health
│     ├─ stt.py               # POST /stt (Whisper STT)
│     ├─ tts.py               # POST /tts (Bark TTS)
│     ├─ sessions.py          # CRUD-ish, in-memory sessions
│     ├─ eval.py              # POST /eval (basic scoring scaffold)
│     └─ sim.py               # Simulation endpoints (interview/chat)
├─ core/
│  ├─ config.py               # Central settings (env vars, models, etc.)
│  ├─ logging.py              # Logging configuration
│  └─ version.py              # App name/version
├─ lifespan.py                # Startup/shutdown hooks (warm GPU models)
├─ main.py                    # FastAPI app; mounts routers; enables CORS
├─ models/
│  ├─ db.py                   # SQLAlchemy engine + session factory
│  └─ session.py              # Session ORM model
├─ schemas/                   # Pydantic v2 request/response contracts
│  ├─ common.py               # Generic Msg schema
│  ├─ ipa.py                  # IPA/phonetic schemas
│  ├─ session.py              # SessionCreate, SessionOut
│  ├─ sim.py                  # Simulation-related schemas
│  ├─ stt.py                  # STTRequest, STTResponse
│  └─ tts.py                  # TTSRequest
├─ services/                  # Business logic (heavy lifting)
│  ├─ stt/whisper_service.py  # GPU faster-whisper STT service
│  ├─ tts/bark_service.py     # Bark GPU TTS service
│  ├─ ipa/                    # IPA generation & conversion
│  ├─ judge/                  # Pronunciation/accuracy judging
│  └─ eval/                   # Scoring/evaluation logic
├─ tools/
│  └─ freeze_bark_voice.py    # Utility to freeze/export Bark voices
└─ utils/
   ├─ audio.py                # Audio helpers (resampling, etc.)
   ├─ idempotency.py          # Request idempotency helpers
   └─ text.py                 # Text cleaning & normalization

web/                          # React frontend (Vite)
├─ public/                    # Static assets
│  └─ styles.css              # Global CSS
├─ src/
│  ├─ api.js                  # API client wrapper
│  ├─ App.jsx                 # Main React app
│  ├─ components/             # UI components (Bubble, ControlBar, etc.)
│  ├─ hooks/                  # Custom hooks (e.g., useRecorder)
│  ├─ libs/                   # Shared libs/utilities
│  └─ utils/                  # Misc frontend helpers
├─ package.json               # Frontend dependencies
├─ vite.config.js             # Vite dev server config
└─ README.md                  # Frontend-specific docs

html/                         # Minimal static demo page
│  ├─ index.html
│  └─ styles.css

tests/                        # Pytest suite
├─ conftest.py                # Shared fixtures
├─ test_gpu.py                # GPU availability tests
├─ test_health.py              # Health check tests
├─ test_sessions.py            # Session endpoints tests
└─ test_sim.py                 # Simulation tests

certs/                        # Local HTTPS certs (Vite dev server)
│  ├─ server.crt
│  └─ server.key

docker-compose.yml            # Docker compose (multi-service setup)
Dockerfile                    # Container build
pyproject.toml                # Poetry/PDM project metadata
requirements.txt              # Python dependencies
uvicorn.ini                   # Uvicorn config
README.md                     # Project docs
```


**Separation of concerns**
- **Routers** = thin HTTP boundary: parse inputs, call services, return schemas.  
- **Services** = heavy business logic: faster-whisper (STT), Bark (TTS), Eval, IPA, Judge.  
- **Schemas** = strong I/O contracts (Pydantic v2): request/response validation, automatic docs.  
- **Core** = shared config, logging, versioning.  
- **Models** = database entities (sessions, users — optional until persistence is wired).  
- **Utils** = common helpers (audio normalization, text cleaning, idempotency).  
- **Tools** = scripts/utilities (e.g., freezing Bark voices).  
- **Frontend (web/)** = React/Vite app that interacts with the API.  


---
## ⚙️ Configuration

`app/core/config.py` defines env-driven settings using Pydantic `BaseSettings`:

| Setting             | Default                     | Purpose |
|---------------------|-----------------------------|---------|
| `ENV`               | `dev`                       | Environment label |
| `API_PREFIX`        | `/api`                      | Prefix applied to feature routers (`stt`, `tts`, `eval`, `sessions`, `sim`) |
| `CORS_ORIGINS`      | `["*"]`                     | Allowed origins for CORS |
| `LOG_LEVEL`         | `INFO`                      | Log level |
| `DATABASE_URL`      | `sqlite:///./english.db`    | SQLAlchemy connection string |
| `WHISPER_MODEL`     | `base`                      | faster-whisper model name |
| `TTS_ENGINE`        | `bark`                      | TTS engine: `"bark"` or `"piper"` |
| `PIPER_VOICE`       | `en_US-amy-medium.onnx`     | Piper voice (legacy / optional) |
| `BARK_DEVICE`       | `cpu`                       | Device for Bark: `"cuda"` or `"cpu"` |
| `BARK_VOICE`        | `v2/en_speaker_6`           | Default Bark voice preset |
| `BARK_TEXT_TEMP`    | `0.7`                       | Bark text generation temperature |
| `BARK_WAVEFORM_TEMP`| `0.7`                       | Bark waveform generation temperature |

Create a `.env` in the project root to override any setting.

**CORS** is enabled in `main.py`.  
- All feature routers are mounted at `{{API_PREFIX}}` (default `/api`).  
- The **health** router is mounted without a prefix (`/health`).  


## 🔌 API Endpoints (current)

> Endpoints below are active in the repo. STT uses **faster-whisper** (GPU-ready),  
> TTS uses **Bark** by default (with optional Piper fallback), and other modules are evolving.

### Health
```
GET /health
→ { "name": "english-trainer-api", "version": "0.2.0", "ok": true }
```

### STT — Speech to Text (Whisper / faster-whisper)
```
POST {{API_PREFIX}}/stt
Content-Type: multipart/form-data
- audio: file (required, wav/mp3)
- payload: optional JSON (STTRequest)

→ 200 OK
{ "text": "hello world", "duration_sec": 2.93, "confidence": 0.97 }
```
Implementation: `app/services/stt/whisper_service.py` using **faster-whisper**.  
Supports mono/16k normalization and GPU execution.

### TTS — Text to Speech (Bark GPU by default)
```
POST {{API_PREFIX}}/tts
Content-Type: application/json
{ "text": "Hello world", "voice": "v2/en_speaker_6" }

→ 200 OK (StreamingResponse)
(binary audio, media_type="audio/wav")
```
Implementation: `app/services/tts/bark_service.py`.  
- Default: Bark (GPU or CPU).  
- Fallback: Piper (if `TTS_ENGINE=piper`).  
- Configurable via `.env` (`BARK_VOICE`, `TTS_ENGINE`).  

### Sessions (in-memory for now)
```
POST   {{API_PREFIX}}/sessions
{ "user_id": 123 }
→ { "id": 1, "user_id": 123, "status": "active" }

GET    {{API_PREFIX}}/sessions
→ [ { "id": 1, "user_id": 123, "status": "active" }, ... ]

POST   {{API_PREFIX}}/sessions/{{sid}}/end
→ { "id": 1, "user_id": 123, "status": "ended" }
```
Currently stored in an in-memory dict (stateless).  
Planned: persist via SQLAlchemy (`models/session.py`).

### Eval (scoring scaffold)
```
POST {{API_PREFIX}}/eval
{ "text": "Hello" }
→ { "score": 0.85 }
```
Stubbed evaluation service (`app/services/eval/`).  
Future: pronunciation accuracy, CEFR grading, detailed feedback.

### Simulation (experimental)
```
POST {{API_PREFIX}}/sim/ask
{ "prompt": "Tell me about yourself" }

→ { "reply": "I’m an AI interviewer. Can you describe your last project?" }
```
Routers under `sim` provide conversation/interview flows, backed by LLMs.  
Work in progress: combine **STT + TTS + judge** for real-time speaking practice.  

## ▶️ Run locally

```bash
# 1. Create and activate virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install backend dependencies
pip install -r requirements.txt

# 3. Run the FastAPI backend (with HTTPS for local dev)
uvicorn app.main:app --reload   --host 0.0.0.0 --port 8000   --ssl-keyfile=cerd/localhost+1-key.pem   --ssl-certfile=cerd/localhost+1.pem

# 4. Health check
curl https://localhost:8000/health --insecure

# 5. API docs
# Open in browser:
#   https://localhost:8000/docs
```

**GPU note:**  
If CUDA is available, both faster-whisper and Bark run on GPU by default.  
Override with `.env` (e.g., `BARK_DEVICE=cpu`, `WHISPER_MODEL=base`).  

**Frontend (React app):**
```bash
cd web
npm install   # or pnpm / yarn
npm run dev   # starts on https://trainer.local:5173
```
Make sure backend (`uvicorn`) is running, then open:  
👉 https://trainer.local:5173  

**Uvicorn options** are also available in `uvicorn.ini` (host, port, reload, workers).

---

## 🧪 Tests

```bash
pytest -q
```

- `tests/test_health.py`: backend liveness check  
- `tests/test_sessions.py`: sessions API  
- `tests/test_eval.py`: eval scaffold  
- `tests/test_gpu.py`: verifies GPU availability  
- `tests/test_sim.py`: simulation routes  

Tip: keep **API tests** lightweight and write **unit tests for services** (Whisper, Bark, Eval) where the heavy logic lives.

---

## 🛣️ Roadmap / Next steps

✅ **STT**: faster-whisper integration (GPU-ready, mono/16k normalization).  
✅ **TTS**: Bark GPU support (voices, temperature controls).  
⬜ **Sessions**: wire to DB model (`models/session.py`) using `models/db.py` engine.  
⬜ **IPA**: add `/ipa/pronounce` (English→IPA→LatAm IPA + respelling).  
⬜ **Pronunciation scoring**: `/pron/score` (audio→phoneme accuracy vs expected text).  
⬜ **Interview simulator**: `/sim/*` (LLM interviewer & judge, using Ollama or GPT).  
⬜ **Lifespan**: preload Bark & Whisper models in `lifespan.py` for faster responses.  
⬜ **Frontend polish**: expand web UI for evaluation feedback & scoring dashboards.  

---

## 🧩 FastAPI concepts in this repo

- **Routers** (`APIRouter`) to group endpoints by feature.  
- **Services** encapsulate business logic (Whisper STT, Bark TTS, Eval, IPA, Judge).  
- **Schemas** (Pydantic v2) define request/response contracts and docs.  
- **Dependencies** (`api/deps.py`) for shared resources.  
- **Lifespan** (startup/shutdown) for GPU model warm-ups.  
- **CORS & settings** centralized in `main.py` and `core/config.py`.  
- **Frontend (React/Vite)** consumes the API via `src/api.js`.  

---

## 🔧 Replace / extend services (guidance)

**WhisperService** (STT):  
- Uses `faster-whisper` to load `{{settings.WHISPER_MODEL}}`.  
- Normalizes audio (mono, 16k).  
- Returns `{ text, duration_sec, confidence }`.  

**BarkService** (TTS):  
- Calls Bark with `{{settings.BARK_VOICE}}`.  
- Streams audio chunks via `StreamingResponse`.  
- Configurable with `.env` (temperature, voice, device).  

**Sessions**:  
- Replace in-memory `_sessions` with SQLAlchemy CRUD.  
- Add `GET /sessions/{id}` and pagination support.  

**Judge / Eval / IPA**:  
- Extend `services/judge` for pronunciation scoring.  
- Add `/ipa/pronounce` endpoints for IPA + respelling.  
- Wire evaluation results into frontend dashboards.  

## 🔐 SSL setup for LAN

To run the backend and frontend securely over your LAN, you’ll need to generate and trust local SSL certificates.  
This allows endpoints like `https://trainer.local:5173` to be accessed from multiple devices on the same network.

### 1. Generate certificates
You can use [`mkcert`](https://github.com/FiloSottile/mkcert) (recommended) or OpenSSL.

**Using mkcert:**
```bash
# Install mkcert and local CA
sudo apt install libnss3-tools
brew install mkcert   # (on macOS)
mkcert -install

# Generate certificates for trainer.local and localhost
mkcert trainer.local localhost 127.0.0.1 ::1
```

This produces files like:
```
trainer.local+2.pem      # certificate
trainer.local+2-key.pem  # private key
```

Place them inside the `certs/` or `cerd/` folder in the repo.

### 2. Configure backend (Uvicorn)
Run FastAPI with SSL enabled:
```bash
uvicorn app.main:app --reload   --host 0.0.0.0 --port 8000   --ssl-keyfile=cerd/localhost+1-key.pem   --ssl-certfile=cerd/localhost+1.pem
```

### 3. Configure frontend (Vite)
In `vite.config.js`, point to the same certs:
```js
server: {
  host: true,
  port: 5173,
  https: {
    key: fs.readFileSync(path.join(certDir, "./server.key")),
    cert: fs.readFileSync(path.join(certDir, "./server.crt")),
  },
  origin: "https://trainer.local:5173",
  hmr: {
    host: "trainer.local",
    protocol: "wss",
    port: 5173,
  },
}
```

### 4. Update hosts file
On each device in your LAN, add this line to `/etc/hosts` (Linux/macOS) or  
`C:\Windows\System32\drivers\etc\hosts` (Windows):

```
192.168.0.xxx trainer.local
```

Replace `192.168.0.xxx` with the IP of the machine running the backend/frontend.

### 5. Access over LAN
Now you can visit:
- Backend: `https://trainer.local:8000/docs`
- Frontend: `https://trainer.local:5173`