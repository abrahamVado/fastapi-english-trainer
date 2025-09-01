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
        {
          role: "user",
          type: "text",
          payload: "What’s a friendly way to start small talk at a meetup?",
          ts: Date.now() - 60000,
        },
        {
          role: "bot",
          type: "text",
          payload: "Try: “What brought you here today?” Then follow with a related experience.",
          ts: Date.now(),
        },
      ],
    },
  ]);
  const currentChatId = chats[0].id;
  const conversationRef = useRef(null);

  const addMessage = useCallback((role, type, payload) => {
    setChats((prev) =>
      prev.map((c) =>
        c.id === currentChatId
          ? { ...c, messages: [...c.messages, { role, type, payload, ts: Date.now() }] }
          : c
      )
    );
  }, [currentChatId]);

  const scrollToBottom = useCallback(() => {
    const el = conversationRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, []);
  useEffect(scrollToBottom, [chats, scrollToBottom]);

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
        addMessage("bot", "text", data?.answer || "Received response.");
        setStatus({ kind: "", text: "Idle — ready" });
      } catch (e) {
        console.error(e);
        addMessage("bot", "text", "Error: failed to send or receive. Try again.");
        setStatus({ kind: "err", text: "Network error" });
      } finally {
        setIsProcessing(false);
      }
    })();
  }, []);

  const { isRecording, isProcessing, setIsProcessing, start, stop } =
    useRecorder({ maxSeconds: 60, onBlob });

  useEffect(() => {
    if (isRecording) setStatus({ kind: "warn", text: "Recording… click stop to send" });
    else if (!isProcessing) setStatus((s) => ({ ...s, kind: "", text: "Idle — ready" }));
  }, [isRecording, isProcessing]);

  const handleRecordClick = () => {
    if (isRecording) stop();
    else start();
  };

  const currentMessages = useMemo(
    () => chats.find((c) => c.id === currentChatId)?.messages || [],
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
