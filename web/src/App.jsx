// src/App.jsx
/**
 * App.jsx — Main UI for your English Trainer / Voice Assistant
 * Flow:
 *  - Record → upload audio (sim/answer/audio)
 *  - LLM reply → speak via TTS (tts/say with use_llm=true, using sim store)
 *  - Optional: Score / Next question
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { API } from "./api";
import StatusChip from "./components/StatusChip.jsx";
import Bubble from "./components/Bubble.jsx";
import { useRecorder } from "./hooks/useRecorder.js";
import ControlBar from "./components/ControlBar.jsx";
import { uuid } from "./utils/uuid.js";

// utility: consistent message shape
function makeMsg(role, type, payload) {
  return { role, type, payload, ts: Date.now() };
}

export default function App() {
  // ----------------------------- UI state -----------------------------------
  const [status, setStatus] = useState({ kind: "", text: "Idle — click the mic" });

  const [chats, setChats] = useState(() => [
    {
      id: uuid(),
      title: "Conversation",
      createdAt: Date.now(),
      messages: [
        makeMsg("user", "text", "What’s a friendly way to start small talk at a meetup?"),
        makeMsg("bot", "text", "Try: “What brought you here today?” Then follow with a related experience."),
      ],
    },
  ]);
  const currentChatId = chats[0].id;

  // simulation/session state
  const [sessionId, setSessionId] = useState(null);
  const [questionId, setQuestionId] = useState(null);
  const [question, setQuestion] = useState("");
  const [lastScores, setLastScores] = useState(null);

  // simple fixed context (keep it in sync with your backend expectations)
  const ROLE  = "developer";
  const LEVEL = "senior";
  const MODE  = "interview";

  // anti-duplicate guards
  const onBlobLock = useRef(false);
  const reqIdRef = useRef(null);

  // auto-scroll
  const conversationRef = useRef(null);

  const addMessage = useCallback(
    (role, type, payload) => {
      setChats((prev) =>
        prev.map((c) =>
          c.id === currentChatId
            ? { ...c, messages: [...c.messages, makeMsg(role, type, payload)] }
            : c
        )
      );
    },
    [currentChatId]
  );

  // -------------------------- Backend warmup (optional) ---------------------
  useEffect(() => {
    API.ttsWarm().catch(() => {}); // non-blocking
  }, []);

  // ----------------------------- Recorder -----------------------------------

  const onBlob = useCallback(
    (blob) => {
      (async () => {
        try {
          // 💡 lock immediately to prevent duplicate runs
          if (onBlobLock.current) return;
          onBlobLock.current = true;

          // consistent request id across both requests
          reqIdRef.current =
            (globalThis.crypto && crypto.randomUUID && crypto.randomUUID()) ||
            Math.random().toString(36).slice(2);

          // UI feedback
          addMessage("user", "audio", blob);
          setStatus({ kind: "warn", text: "Transcribing…" });

          let sid = sessionId;
          let qid = questionId;

          // Auto-start if user recorded before hitting Start
          if (!sid || !qid) {
            const s = await API.simStart({ role: ROLE, level: LEVEL, mode: MODE });
            sid = s.session_id;
            qid = s.question_id;
            setSessionId(sid);
            setQuestionId(qid);
            setQuestion(s.question);
            addMessage("bot", "text", s.question);
          }

          // Upload audio for current question → ASR text
          const form = new FormData();
          form.append("session_id", sid);
          form.append("question_id", qid);
          form.append("audio", new File([blob], "answer.webm", { type: blob.type || "audio/webm" }));

          const asrRes = await fetch(`${API.base}/api/sim/answer/audio`, {
            method: "POST",
            headers: { "X-Req-Id": reqIdRef.current },
            body: form,
          });
          if (!asrRes.ok) throw new Error(`sim/answer/audio -> ${asrRes.status}`);
          const asrJson = await asrRes.json();
          const asr = asrJson.asr_text || "(no speech detected)";
          addMessage("user", "text", asr);

          // 🔊 Ask tutor (LLM) based on the recorded answer and speak it via TTS
          setStatus({ kind: "busy", text: "Tutor thinking… then speaking reply…" });
          const ttsRes = await fetch(`${API.base}/api/tts/say`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "X-Req-Id": reqIdRef.current,
            },
            body: JSON.stringify({
              use_llm: true,
              session_id: sid,
              question_id: qid,
              role: ROLE,
              level: LEVEL,
              mode: MODE,
              voice: "v2/en_speaker_7"
            }),
          });
          if (!ttsRes.ok) throw new Error(`tts/say -> ${ttsRes.status}`);
          const wav = await ttsRes.blob();

          // Play returned WAV
          const url = URL.createObjectURL(wav);
          const audio = new Audio(url);
          try {
            await audio.play();
          } catch (err) {
            console.error("Playback blocked; show a play button:", err);
          } finally {
            // keep it around briefly in case of replay UI; then cleanup
            setTimeout(() => URL.revokeObjectURL(url), 30000);
          }

          // show a “bot audio” bubble if your <Bubble> supports it
          addMessage("bot", "audio", wav);

          setStatus({ kind: "", text: "Ready" });
        } catch (e) {
          console.error(e);
          addMessage("bot", "text", "Error: audio upload/LLM-TTS failed.");
          setStatus({ kind: "err", text: "Audio/LLM error" });
        } finally {
          setIsProcessing(false);
          onBlobLock.current = false; // ✅ always release
        }
      })();
    },
    [addMessage, sessionId, questionId]
  );

  const { isRecording, isProcessing, setIsProcessing, start, stop } = useRecorder({
    maxSeconds: 60,
    onBlob,
  });

  useEffect(() => {
    if (isRecording) setStatus({ kind: "warn", text: "Recording… click stop to send" });
    else if (!isProcessing) setStatus((s) => ({ ...s, kind: "", text: "Idle — ready" }));
  }, [isRecording, isProcessing]);

  const handleRecordClick = () => (isRecording ? stop() : start());

  // ----------------------------- Sim controls -------------------------------
  async function handleStart() {
    setStatus({ kind: "busy", text: "Starting session..." });
    try {
      const res = await API.simStart({ role: ROLE, level: LEVEL, mode: MODE });
      setSessionId(res.session_id);
      setQuestionId(res.question_id);
      setQuestion(res.question);
      addMessage("bot", "text", res.question);
      setLastScores(null);
      setStatus({ kind: "", text: "Session ready" });
    } catch (e) {
      console.error(e);
      setStatus({ kind: "err", text: "Start failed" });
    }
  }

  async function handleNext() {
    if (!sessionId) return;
    setStatus({ kind: "busy", text: "Getting next question..." });
    try {
      const res = await API.simNext({ session_id: sessionId });
      setQuestionId(res.question_id);
      setQuestion(res.question);
      addMessage("bot", "text", res.question);
      setLastScores(null);
      setStatus({ kind: "", text: "Next question" });
    } catch (e) {
      console.error(e);
      setStatus({ kind: "err", text: "Next failed" });
    }
  }

  async function handleScore() {
    if (!sessionId || !questionId) return;
    setStatus({ kind: "busy", text: "Scoring..." });
    try {
      const res = await API.simScore({ session_id: sessionId, question_id: questionId });
      setLastScores(res.scores);
      addMessage(
        "bot",
        "text",
        `Overall: ${res.scores.overall} — Tips: ${res.tips.join(" | ")}`
      );
      setStatus({ kind: "", text: "Scored" });
    } catch (e) {
      console.error(e);
      setStatus({ kind: "err", text: "Score failed" });
    }
  }

  // auto-scroll to newest message
  const conversationRefCb = useCallback(
    (node) => {
      conversationRef.current = node;
      if (node) node.scrollTop = node.scrollHeight;
    },
    []
  );
  useEffect(() => {
    const el = conversationRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [chats]);

  const currentMessages = useMemo(
    () => chats.find((c) => c.id === currentChatId)?.messages || [],
    [chats, currentChatId]
  );

  // --------------------------- Render ---------------------------------------
  return (
    <div className="conversation-container conversation-container--no-sidebar">
      <div className="app">
        {/* Header: title + small status indicator */}
        <header className="app-header">
          <div className="logo" aria-hidden="true" />
          <div className="title">English Trainer</div>
          <StatusChip kind={status.kind} text={status.text} />
        </header>

        {/* Chat + controls */}
        <div className="app-body">
          <section className="panel">
            <div className="convo-card">
              <div className="chat-head">
                <div className="title" id="modeTitle">
                  {question ? "Interview Mode" : "Conversation"}
                </div>
                {question && (
                  <div style={{ marginTop: 6 }}>
                    <strong>Question:</strong> {question}
                  </div>
                )}
                {lastScores && (
                  <div style={{ marginTop: 6 }}>
                    <strong>Scores:</strong>{" "}
                    content {lastScores.content}, pron {lastScores.pronunciation}, fluency{" "}
                    {lastScores.fluency}, <strong>overall {lastScores.overall}</strong>
                  </div>
                )}
              </div>

              {/* Chat transcript */}
              <div
                id="conversation"
                role="log"
                aria-live="polite"
                aria-relevant="additions"
                ref={conversationRefCb}
              >
                {currentMessages.map((m, i) => (
                  <Bubble m={m} key={i} />
                ))}
              </div>

              {/* Unified control bar (mic + Start/Next/Score) */}
              <div id="controls" className="controls">
                <ControlBar
                  recording={isRecording}
                  processing={isProcessing}
                  onMicClick={handleRecordClick}
                  onStart={handleStart}
                  onNext={handleNext}
                  onScore={handleScore}
                  canNext={!!sessionId}
                  canScore={!!sessionId && !!questionId}
                />
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
