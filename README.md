# English Trainer API (FastAPI modular)

## Endpoints
- GET /health
- POST /api/stt
- POST /api/tts
- POST /api/eval
- CRUD /api/sessions

## Run
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
