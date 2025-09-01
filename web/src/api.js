// src/api.js
// src/api.js
const base =
  (typeof import.meta !== "undefined" && import.meta.env?.VITE_API_BASE_URL) ||
  "https://localhost:8000";


const u = (p) => `${base}${p}`;

console.log(base);
// -------------------- API methods --------------------

// Health check
async function health() {
  const r = await fetch(u("/health"));
  if (!r.ok) throw new Error(`health failed: ${r.status}`);
  return r.json();
}

// Simulation
async function simStart(payload) {
  const r = await fetch(u("/api/sim/start"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw new Error(`start failed: ${r.status}`);
  return r.json();
}

async function simNext(payload) {
  const r = await fetch(u("/api/sim/next"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw new Error(`next failed: ${r.status}`);
  return r.json();
}

async function simAnswerText(payload) {
  const r = await fetch(u("/api/sim/answer/text"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw new Error(`answer text failed: ${r.status}`);
  return r.json();
}

async function simAnswerAudio(session_id, question_id, blob) {
  const form = new FormData();
  form.append("session_id", session_id);
  form.append("question_id", question_id);
  form.append(
    "audio",
    new File([blob], "answer.webm", { type: blob.type || "audio/webm" })
  );

  const r = await fetch(u("/api/sim/answer/audio"), { method: "POST", body: form });
  if (!r.ok) throw new Error(`answer audio failed: ${r.status}`);
  return r.json();
}

async function simScore(payload) {
  const r = await fetch(u("/api/sim/score"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw new Error(`score failed: ${r.status}`);
  return r.json();
}

async function simReport(session_id) {
  const r = await fetch(u(`/api/sim/report?session_id=${encodeURIComponent(session_id)}`));
  if (!r.ok) throw new Error(`report failed: ${r.status}`);
  return r.json();
}

// -------------------- Export as object --------------------

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
  tts: u("/api/tts"),
  ttsWarm: u("/api/tts/warm"),
};
