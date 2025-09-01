// src/api.js
// Centralized API client for the frontend

// IMPORTANT: set this in web/.env.development (or your Vite env):
//   VITE_API_BASE_URL=https://trainer.local:8000
const base =
  (typeof import.meta !== "undefined" && import.meta.env?.VITE_API_BASE_URL) ||
  "https://trainer.local:8000"; // fallback for dev

const u = (p) => `${base}${p}`;

// ---------- small helpers ----------
async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${opts?.method || "GET"} ${url} -> ${r.status}`);
  return r.json();
}
async function getBlob(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${opts?.method || "GET"} ${url} -> ${r.status}`);
  return r.blob();
}

// -------------------- Health --------------------
async function health() {
  return getJSON(u("/health"));
}

// -------------------- Simulation --------------------
async function simStart(payload) {
  return getJSON(u("/api/sim/start"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

async function simNext(payload) {
  return getJSON(u("/api/sim/next"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

async function simAnswerText(payload) {
  return getJSON(u("/api/sim/answer/text"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

async function simAnswerAudio(session_id, question_id, blob) {
  const form = new FormData();
  form.append("session_id", session_id);
  form.append("question_id", question_id);
  form.append(
    "audio",
    new File([blob], "answer.webm", { type: blob.type || "audio/webm" })
  );

  return getJSON(u("/api/sim/answer/audio"), { method: "POST", body: form });
}

async function simScore(payload) {
  return getJSON(u("/api/sim/score"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

async function simReport(session_id) {
  return getJSON(u(`/api/sim/report?session_id=${encodeURIComponent(session_id)}`));
}

// -------------------- TTS --------------------
// Warmup (optional)
async function ttsWarm() {
  return fetch(u("/api/tts/warm"), { method: "POST" });
}

// Plain say (no LLM): returns a WAV Blob
async function ttsSay(text) {
  return getBlob(u("/api/tts/say"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
}

// LLM->TTS from sim store (use last ASR answer)
async function ttsSayLLMFromSim(session_id, question_id, ctx = {}) {
  const payload = {
    use_llm: true,
    session_id,
    question_id,
    ...ctx, // role, level, mode, llm_model (optional)
  };
  return getBlob(u("/api/tts/say"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

// LLM->TTS from arbitrary text
async function ttsSayLLMFromText(text, ctx = {}) {
  const payload = {
    use_llm: true,
    text,
    ...ctx, // role, level, mode, llm_model (optional)
  };
  return getBlob(u("/api/tts/say"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

// -------------------- Export --------------------
export const API = {
  base,
  // health
  health,
  // sim
  simStart,
  simNext,
  simAnswerText,
  simAnswerAudio,
  simScore,
  simReport,
  // tts
  ttsWarm,
  ttsSay,
  ttsSayLLMFromSim,
  ttsSayLLMFromText,
};
