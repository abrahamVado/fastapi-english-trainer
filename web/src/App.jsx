// src/App.jsx
/**
 * App.jsx — Main UI for your English Trainer / Voice Assistant
 *
 * Uses a single API object (import { API } from "./api") for all backend calls:
 *   - API.health()
 *   - API.simStart(), API.simNext(), API.simAnswerAudio(), API.simScore(), API.simReport()
 *   - API.tts (URL), API.ttsWarm (URL)
 *
 * Flow:
 *  - Record with useRecorder → onBlob gets audio → ensure session → API.simAnswerAudio
 *  - Control bar: Start / Next / Score actions hit the API.* methods
 *  - TTS warm on mount; speak() posts to API.tts
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { API } from "./api";                      // ✅ single import for all endpoints/URLs
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
  const [lastScores, setLastScores] = useState(null); // { content, pronunciation, fluency, overall }

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
    // warm TTS (non-blocking; ignore errors)
    API.ttsWarm().catch(() => {});
    // Optional health check:
    // API.health().then(() => setStatus({ kind: "", text: "Backend OK — ready" })).catch(() => {});
  }, []);

  // ----------------------------- Recorder -----------------------------------
  /**
   * useRecorder:
   * - start(): begin mic capture (browser prompts permission on first use)
   * - stop(): finish → calls our onBlob(blob)
   * - isRecording / isProcessing: control UI states
   */

  // onBlob runs after a recording finishes:
  // - ensure we have a session/question (auto-start if missing)
  // - upload audio to API.simAnswerAudio
  // - add the recognized text to the chat
  const onBlob = useCallback(
    (blob) => {
      addMessage("user", "audio", blob);
      setStatus({ kind: "warn", text: "Transcribing…" });

      (async () => {
        try {
          let sid = sessionId;
          let qid = questionId;

          // Auto-start if user recorded before hitting Start
          if (!sid || !qid) {
            const s = await API.simStart({ role: "developer", level: "senior", mode: "interview" });
            sid = s.session_id;
            qid = s.question_id;
            setSessionId(sid);
            setQuestionId(qid);
            setQuestion(s.question);
            addMessage("bot", "text", s.question);
          }

          // Upload audio for current question
          const res = await API.simAnswerAudio(sid, qid, blob);
          const asr = res.asr_text || "(no speech detected)";
          addMessage("user", "text", asr);
          const textToSpeak = asr;
          // start simple: echo back what you said

          // Warm TTS in background (optional)
          API.ttsWarm().catch(()=>{});

          // Get audio and play it
          const audioBlob = await API.ttsSay(textToSpeak);
          const url = URL.createObjectURL(audioBlob);
          const audio = new Audio(url);
          try {
            await audio.play(); // make sure this runs as a result of a user gesture to avoid autoplay blocking
          } catch (err) {
            console.error("Playback blocked; show a play button:", err);
          }
          setTimeout(() => URL.revokeObjectURL(url), 30000);
          setStatus({ kind: "", text: "Ready" });
        } catch (e) {
          console.error(e);
          addMessage("bot", "text", "Error: audio upload/transcription failed.");
          setStatus({ kind: "err", text: "Audio error" });
        } finally {
          setIsProcessing(false);
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
      const res = await API.simStart({ role: "developer", level: "senior", mode: "interview" });
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
      addMessage("bot", "text", `Overall: ${res.scores.overall} — Tips: ${res.tips.join(" | ")}`);
      setStatus({ kind: "", text: "Scored" });
    } catch (e) {
      console.error(e);
      setStatus({ kind: "err", text: "Score failed" });
    }
  }

  // Speak a bot reply via backend TTS
  const speak = useCallback(
    async (text) => {
      try {
        const res = await fetch(API.tts, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, voice: "v2/es_speaker_0" }),
        });
        if (!res.ok) {
          console.warn("TTS failed", await res.text());
          return;
        }
        const blob = await res.blob(); // e.g. WAV
        addMessage("bot", "audio", blob);
      } catch (e) {
        console.warn("TTS error", e);
      }
    },
    [addMessage]
  );

  // auto-scroll to newest message
  const scrollToBottom = useCallback(() => {
    const el = conversationRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, []);
  useEffect(scrollToBottom, [chats, scrollToBottom]);

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
                    content {lastScores.content}, pron {lastScores.pronunciation}, fluency {lastScores.fluency},{" "}
                    <strong>overall {lastScores.overall}</strong>
                  </div>
                )}
              </div>

              {/* Chat transcript */}
              <div
                id="conversation"
                role="log"
                aria-live="polite"
                aria-relevant="additions"
                ref={conversationRef}
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
