# Voice Assistant — Conversation (React + Vite)

This is a minimal React app to send microphone audio to your Python API and display the conversation. **Agents sidebar and Reader mode were removed** per your request.

## Quick start

```bash
# inside the extracted folder
npm i
npm run dev
```

Open the printed local URL.

## Configure API endpoint

By default, the app POSTs to `/api/audio`:

- Create a `.env` file to override:
```
VITE_API_AUDIO=http://localhost:8000/api/audio
```

Restart the dev server after changing envs.

## How it works

- Click the mic (or hold **Space**) to record; release to stop & send.
- Audio is sent as `multipart/form-data` with fields:
  - `file`: the recorded audio blob (WebM/Opus or OGG/Opus depending on browser support).
  - `mode`: `"conversation"` (for parity with your backend).
- The response is expected to be JSON: `{ "answer": "..." }`.
- Messages (user audio + bot text) render as bubbles.

## Files of interest

- `src/hooks/useRecorder.js` — MediaRecorder + push‑to‑talk logic
- `src/App.jsx` — conversation-only UI
- `public/styles.css` — drop your existing stylesheet here (class names preserved)
