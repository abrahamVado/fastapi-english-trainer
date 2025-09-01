import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { API } from "./api";
import StatusChip from "./components/StatusChip.jsx";
import RecordButton from "./components/RecordButton.jsx";
import Bubble from "./components/Bubble.jsx";
import { useRecorder } from "./hooks/useRecorder.js";

export default function App() {
  const [status, setStatus] = useState({ kind: "", text: "Idle — click the mic" });
  const [chats, setChats] = useState(() => [
    {
      id: crypto.randomUUID(),
      title: "Conversation",
      createdAt: Date.now(),
      messages: [
        { role: "user", type: "text", payload: "What’s a friendly way to start small talk at a meetup?", ts: Date.now()-60000 },
        { role: "bot",  type: "text", payload: "Try: “What brought you here today?” Then follow with a related experience.", ts: Date.now() },
      ],
    },
  ]);
  const currentChatId = chats[0].id;
  const conversationRef = useRef(null);

  const addMessage = useCallback((role, type, payload) => {
    setChats(prev =>
      prev.map(c =>
        c.id === currentChatId
          ? { ...c, messages: [...c.messages, { role, type, payload, ts: Date.now() }] }
          : c
      )
    );
  }, [currentChatId]);

  // Warm Bark models on mount (non-blocking)
  useEffect(() => {
    fetch(API.ttsWarm, { method: "POST" }).catch(() => {});
  }, []);

  // recorder FIRST (so setIsProcessing exists when onBlob runs)
  const { isRecording, isProcessing, setIsProcessing, start, stop } =
    useRecorder({ maxSeconds: 60, onBlob: null });

  const scrollToBottom = useCallback(() => {
    const el = conversationRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, []);
  useEffect(scrollToBottom, [chats, scrollToBottom]);

  // speak helper → call Bark and append audio
  const speak = useCallback(async (text) => {
    try {
      const res = await fetch(API.tts, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          voice: "v2/es_speaker_0", // pick your default voice
          // use_small: true,
          // seed: 1234,
        }),
      });
      if (res.ok) {
        const blob = await res.blob();          // audio/wav
        addMessage("bot", "audio", blob);
      } else {
        console.warn("TTS failed", await res.text());
      }
    } catch (e) {
      console.warn("TTS error", e);
    }
  }, [addMessage]);

  // now define onBlob using the recorder
  const onBlob = useCallback((blob) => {
    addMessage("user", "audio", blob);
    setStatus({ kind: "warn", text: "Sending… please wait" });
    (async () => {
      try {
        const form = new FormData();
        form.append("file", blob, "message.webm");
        form.append("mode", "conversation");
        const r = await fetch(API.audio, { method: "POST", body: form });
        const data = await r.json().catch(() => ({}));
        const answer = data?.answer || "Received response.";

        addMessage("bot", "text", answer);
        speak(answer); // 🔊 give the bot a voice
        setStatus({ kind: "", text: "Idle — ready" });
      } catch (e) {
        console.error(e);
        addMessage("bot", "text", "Error: failed to send or receive. Try again.");
        setStatus({ kind: "err", text: "Network error" });
      } finally {
        setIsProcessing(false);
      }
    })();
  }, [addMessage, setIsProcessing, speak]);

  // wire onBlob into the recorder (since we created the hook earlier)
  useEffect(() => {
    // quick way to “rebind” the callback without changing the hook signature:
    // start/stop unaffected; we just need onBlob to be current.
    (window).__onBlob = onBlob; // debug convenience
  }, [onBlob]);

  // update status by recording state
  useEffect(() => {
    if (isRecording) setStatus({ kind: "warn", text: "Recording… click stop to send" });
    else if (!isProcessing) setStatus(s => ({ ...s, kind: "", text: "Idle — ready" }));
  }, [isRecording, isProcessing]);

  const handleRecordClick = () => { isRecording ? stop() : start(onBlob); };
  // If your useRecorder expects the callback in options only, adjust hook to accept start(cb)

  const currentMessages = useMemo(
    () => chats.find(c => c.id === currentChatId)?.messages || [],
    [chats, currentChatId]
  );

  return (
    <div className="conversation-container conversation-container--no-sidebar">
      <div className="app">
        <header className="app-header">
          <div className="logo" aria-hidden="true" />
          <div className="title">Voice Assistant</div>
          <StatusChip kind={status.kind} text={status.text} />
        </header>

        <div className="app-body">
          <section className="panel">
            <div className="convo-card">
              <div className="chat-head">
                <div className="title" id="modeTitle">Conversation</div>
              </div>

              <div id="conversation" role="log" aria-live="polite" aria-relevant="additions" ref={conversationRef}>
                {currentMessages.map((m, i) => <Bubble m={m} key={i} />)}
              </div>

              <div id="controls" className="controls">
                <RecordButton recording={isRecording} onClick={handleRecordClick} />
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
