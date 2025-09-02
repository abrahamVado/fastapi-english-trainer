// src/hooks/useRecorder.js
import { useCallback, useEffect, useRef, useState } from "react";

const MIME_CANDIDATES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/ogg;codecs=opus",
  "audio/ogg",
];

function pickSupportedMime() {
  for (const m of MIME_CANDIDATES) {
    try { if (window.MediaRecorder?.isTypeSupported?.(m)) return m; } catch {}
  }
  return "";
}

function hasSecureMicSupport() {
  if (window.isSecureContext) return true;
  return /^(localhost|127\.0\.0\.1)$/.test(location.hostname);
}

export function useRecorder({
  maxSeconds = 60,
  onBlob,
  enableHotkeys = true,
  autoStopOnBlur = true,
  audioConstraints,
  deviceId,
} = {}) {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [mimeType, setMimeType] = useState("");
  const [durationSec, setDurationSec] = useState(0);
  const [permission, setPermission] = useState("unknown");

  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);
  const startedAtRef = useRef(0);

  // NEW: guard against rapid double-starts before state updates land
  const startLockRef = useRef(false);
  // NEW: per-recording session id to ignore stale events
  const sessionIdRef = useRef(0);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        if (navigator.permissions?.query) {
          const st = await navigator.permissions.query({ name: "microphone" });
          if (!cancelled) {
            setPermission(st.state === "prompt" ? "unknown" : st.state);
            st.onchange = () => setPermission(st.state === "prompt" ? "unknown" : st.state);
          }
        }
      } catch {}
    })();
    return () => { cancelled = true; };
  }, []);

  // Choose a MIME once
  useEffect(() => {
    setMimeType(pickSupportedMime() || "(browser default)");
  }, []);

  // Stop tracks & timers; DO NOT call mr.stop() here (can re-enter onstop)
  const teardown = useCallback(() => {
    try { streamRef.current?.getTracks?.().forEach((t) => t.stop()); } catch {}
    streamRef.current = null;

    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const start = useCallback(async () => {
    if (isRecording || isProcessing || startLockRef.current) return;
    startLockRef.current = true;
    setError(null);

    if (!hasSecureMicSupport()) {
      const msg = "Mic requires HTTPS or http://localhost.";
      setError(msg);
      startLockRef.current = false;
      throw new Error(msg);
    }

    // Polyfill getUserMedia if needed (legacy)
    if (!navigator.mediaDevices?.getUserMedia) {
      const legacy = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
      if (legacy) {
        navigator.mediaDevices = navigator.mediaDevices || {};
        navigator.mediaDevices.getUserMedia = (c) =>
          new Promise((res, rej) => legacy.call(navigator, c, res, rej));
      } else {
        const msg = "getUserMedia not available. Use a modern browser over HTTPS or localhost.";
        setError(msg);
        startLockRef.current = false;
        throw new Error(msg);
      }
    }

    try {
      const constraints = {
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          ...(audioConstraints || {}),
          ...(deviceId ? { deviceId } : {}),
        },
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      setPermission("granted");
      streamRef.current = stream;

      const chosen = pickSupportedMime();
      const mr = new MediaRecorder(stream, chosen ? { mimeType: chosen } : undefined);

      chunksRef.current = [];
      const mySession = ++sessionIdRef.current; // NEW session id

      mr.ondataavailable = (e) => {
        if (sessionIdRef.current !== mySession) return; // ignore stale
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
      };

      mr.onstop = async () => {
        if (sessionIdRef.current !== mySession) return; // ignore stale
        const elapsed = Math.max(0, Math.round((performance.now() - startedAtRef.current) / 1000));
        setDurationSec(elapsed);

        const blob = new Blob(chunksRef.current, { type: mr.mimeType || "audio/webm" });
        chunksRef.current = [];

        // Clear recorder ref BEFORE teardown to prevent re-entry
        mediaRecorderRef.current = null;
        setIsRecording(false);

        try {
          if (blob.size && typeof onBlob === "function") {
            setIsProcessing(true);
            await onBlob(blob);
          }
        } catch (err) {
          console.error("onBlob error:", err);
          setError(err?.message || String(err));
        } finally {
          setIsProcessing(false);
          teardown();
        }
      };

      mediaRecorderRef.current = mr;
      mr.start();                 // one-shot; no timeslice → single final blob
      startedAtRef.current = performance.now();
      setIsRecording(true);

      if (maxSeconds && maxSeconds > 0) {
        timerRef.current = setTimeout(() => {
          // guard double stops
          if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
            try { mediaRecorderRef.current.stop(); } catch {}
          }
        }, maxSeconds * 1000);
      }
    } catch (err) {
      console.error(err);
      setError(err?.message || String(err));
      setPermission(err?.name === "NotAllowedError" ? "denied" : "unknown");
      setIsRecording(false);
      setIsProcessing(false);
      teardown();
      startLockRef.current = false;
      throw err;
    } finally {
      // Release the start lock a tick later to avoid back-to-back clicks
      setTimeout(() => { startLockRef.current = false; }, 0);
    }
  }, [isRecording, isProcessing, maxSeconds, onBlob, audioConstraints, deviceId, teardown]);

  const stop = useCallback(() => {
    const mr = mediaRecorderRef.current;
    if (!mr || mr.state !== "recording") return;
    try { mr.stop(); } catch {}
    // isRecording will be set false in onstop after the blob is built
    if (timerRef.current) { clearTimeout(timerRef.current); timerRef.current = null; }
  }, []);

  const cancel = useCallback(() => {
    const mr = mediaRecorderRef.current;
    chunksRef.current = [];
    if (mr && mr.state === "recording") {
      try { mr.stop(); } catch {}
    }
    mediaRecorderRef.current = null;
    setIsRecording(false);
    setIsProcessing(false);
    teardown();
  }, [teardown]);

  // Hotkeys: Space (hold-to-record), Esc (cancel)
  useEffect(() => {
    if (!enableHotkeys) return;

    const down = (e) => {
      if (e.code === "Space" && !e.repeat) {
        e.preventDefault();
        if (!isRecording && !isProcessing) start();
      } else if (e.code === "Escape") {
        e.preventDefault();
        if (isRecording) cancel();
      }
    };
    const up = (e) => {
      if (e.code === "Space") {
        e.preventDefault();
        if (isRecording) stop();
      }
    };

    document.addEventListener("keydown", down);
    document.addEventListener("keyup", up);
    return () => {
      document.removeEventListener("keydown", down);
      document.removeEventListener("keyup", up);
    };
  }, [enableHotkeys, isRecording, isProcessing, start, stop, cancel]);

  // Auto-stop on tab hide
  useEffect(() => {
    if (!autoStopOnBlur) return;
    const onVis = () => {
      if (document.hidden && mediaRecorderRef.current?.state === "recording") {
        try { mediaRecorderRef.current.stop(); } catch {}
      }
    };
    document.addEventListener("visibilitychange", onVis);
    return () => document.removeEventListener("visibilitychange", onVis);
  }, [autoStopOnBlur]);

  // Cleanup on unmount
  useEffect(() => teardown, [teardown]);

  return {
    isRecording,
    isProcessing,
    error,
    mimeType,
    durationSec,
    permission,
    start,
    stop,
    cancel,
    setIsProcessing,
  };
}
