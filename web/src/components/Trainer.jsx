// src/components/Trainer.jsx
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { API } from "../api";
import { recordOnce } from "../lib/recorder";
import { playBlob } from "../lib/audio";

export default function Trainer() {
  const [session, setSession] = useState(null); // { session_id, question_id, question }
  const [answering, setAnswering] = useState(false);
  const [status, setStatus] = useState("");
  const [role, setRole] = useState("backend developer");
  const [level, setLevel] = useState("intermediate");
  const [mode, setMode] = useState("interview");

  useEffect(() => {
    // optional warmup
    API.ttsWarm().catch(() => {});
  }, []);

  const startSession = useCallback(async () => {
    setStatus("Starting session...");
    const res = await API.simStart({ role, level, mode });
    setSession(res); // { session_id, question_id, question }
    setStatus("Session started. Ask/answer the question!");
  }, [role, level, mode]);

  const recordAndSend = useCallback(async () => {
    if (!session) return;
    setAnswering(true);
    setStatus("Recording (up to 8s)...");
    try {
      const blob = await recordOnce({ maxMs: 8000 });
      setStatus("Uploading audio...");
      await API.simAnswerAudio(session.session_id, session.question_id, blob);
      setStatus("Got ASR. Asking tutor (LLM) and speaking reply...");
      const wav = await API.ttsSayLLMFromSim(session.session_id, session.question_id, {
        role,
        level,
        mode,
        // llm_model: "llama3.1", // optional override
      });
      await playBlob(wav);
      setStatus("Tutor reply played ✅");
    } catch (e) {
      console.error(e);
      setStatus("Error: " + e.message);
    } finally {
      setAnswering(false);
    }
  }, [session, role, level, mode]);

  const askNext = useCallback(async () => {
    if (!session) return;
    setStatus("Fetching next question...");
    const res = await API.simNext({ session_id: session.session_id });
    setSession((s) => ({ ...s, question_id: res.question_id, question: res.question }));
    setStatus("New question ready.");
  }, [session]);

  const scoreLast = useCallback(async () => {
    if (!session) return;
    setStatus("Scoring last answer...");
    const res = await API.simScore({
      session_id: session.session_id,
      question_id: session.question_id,
    });
    setStatus(
      `Scores — content: ${res.scores.content}, pron: ${res.scores.pronunciation}, ` +
      `fluency: ${res.scores.fluency}, overall: ${res.scores.overall}`
    );
  }, [session]);

  return (
    <div style={{ maxWidth: 760, margin: "40px auto", fontFamily: "system-ui, sans-serif" }}>
      <h1>English Trainer</h1>
      <p style={{ opacity: 0.8 }}>
        API: <code>{API.base}</code>
      </p>

      <fieldset style={{ margin: "16px 0", padding: 12 }}>
        <legend>Context</legend>
        <label style={{ display: "block", marginBottom: 8 }}>
          Role:{" "}
          <input value={role} onChange={(e) => setRole(e.target.value)} placeholder="backend developer" />
        </label>
        <label style={{ display: "block", marginBottom: 8 }}>
          Level:{" "}
          <input value={level} onChange={(e) => setLevel(e.target.value)} placeholder="intermediate" />
        </label>
        <label style={{ display: "block" }}>
          Mode:{" "}
          <input value={mode} onChange={(e) => setMode(e.target.value)} placeholder="interview" />
        </label>
      </fieldset>

      {!session ? (
        <button onClick={startSession} style={{ padding: "10px 16px" }}>
          Start Session
        </button>
      ) : (
        <>
          <div style={{ marginTop: 12, padding: 12, background: "#f7f7f9", borderRadius: 8 }}>
            <div style={{ fontWeight: 600, marginBottom: 6 }}>Question</div>
            <div>{session.question}</div>
            <div style={{ marginTop: 8, fontSize: 12, opacity: 0.7 }}>
              session_id: <code>{session.session_id}</code>
              <br />
              question_id: <code>{session.question_id}</code>
            </div>
          </div>

          <div style={{ marginTop: 16, display: "flex", gap: 8 }}>
            <button onClick={recordAndSend} disabled={answering} style={{ padding: "10px 16px" }}>
              {answering ? "Recording..." : "Answer (record 8s) + Tutor reply"}
            </button>
            <button onClick={scoreLast} style={{ padding: "10px 16px" }}>
              Score last answer
            </button>
            <button onClick={askNext} style={{ padding: "10px 16px" }}>
              Next question
            </button>
          </div>
        </>
      )}

      <div style={{ marginTop: 18, color: "#444" }}>{status}</div>
    </div>
  );
}
