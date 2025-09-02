// src/hooks/useRecorder.js
import { useCallback, useEffect, useRef, useState } from "react";

const MIME_CANDIDATES = [
  "audio/webm;codecs=opus",         // Chrome/Edge/Firefox
  "audio/webm",
  "audio/ogg;codecs=opus",          // Firefox alt
  "audio/ogg",
  "audio/mp4;codecs=mp4a.40.2",     // Safari (modern)
  "audio/mp4",                       // Safari fallback
];

function pickSupportedMime() {
  for (const m of MIME_CANDIDATES) {
    try { if (window.MediaRecorder?.isTypeSupported?.(m)) return m; } catch {}
  }
  return ""; // let browser pick
}

function hasSecureMicSupport() {
  if (window.isSecureContext) return true;
  // Allow localhost and .local dev
  if (/^(localhost|127\.0\.0\.1|::1|.*\.local)$/i.test(location.hostname)) return true;
  return false;
}

/**
 * useRecorder — robust voice recorder with:
 * - AEC/NS/AGC constraints
 * - client-side VAD auto-stop
 * - level metering via onLevel(db)
 * - chunk streaming via onChunk (optional) or final onBlob
 */
export function useRecorder({
  maxSeconds = 60,
  onBlob,               // async (blob) => void
  onChunk,              // async ({blob,isLast}) => void  (optional)
  onLevel,              // (db) => void (optional, for meters)
  enableHotkeys = true,
  autoStopOnBlur = true,
  audioConstraints,
  deviceId,
  vad = { enabled: true, silenceMs: 1200, thresholdDb: -45, minStartMs: 250 },
  timesliceMs = 0,      // if >0, MediaRecorder will emit chunks periodically
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

  // NEW: lock & session guards
  const startLockRef = useRef(false);
  const sessionIdRef = useRef(0);

  // NEW: Web Audio nodes for VAD/meters
  const audioCtxRef = useRef(null);
  const analyserRef = useRef(null);
  const sourceNodeRef = useRef(null);
  const rafRef = useRef(null);
  const silenceStartRef = useRef(null);

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

  useEffect(() => { setMimeType(pickSupportedMime() || "(browser default)"); }, []);

  const teardown = useCallback(() => {
    try { streamRef.current?.getTracks?.().forEach((t) => t.stop()); } catch {}
    streamRef.current = null;

    if (timerRef.current) { clearTimeout(timerRef.current); timerRef.current = null; }

    // WebAudio cleanup
    try {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      if (sourceNodeRef.current) sourceNodeRef.current.disconnect();
      sourceNodeRef.current = null;
      if (analyserRef.current) analyserRef.current.disconnect();
      analyserRef.current = null;
      if (audioCtxRef.current) {
        audioCtxRef.current.close().catch(()=>{});
      }
    } catch {}
    audioCtxRef.current = null;
    silenceStartRef.current = null;
  }, []);

  // simple RMS→dB
  const _levelDb = (pcm) => {
    let sum = 0;
    for (let i = 0; i < pcm.length; i++) { const v = pcm[i]; sum += v*v; }
    const rms = Math.sqrt(sum / (pcm.length || 1));
    const db = 20 * Math.log10(rms + 1e-7);
    return Math.max(-100, Math.min(0, db));
  };

  const _startMetering = useCallback((stream) => {
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      audioCtxRef.current = ctx;
      const src = ctx.createMediaStreamSource(stream);
      sourceNodeRef.current = src;

      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      analyser.smoothingTimeConstant = 0.2;
      analyserRef.current = analyser;

      src.connect(analyser);

      const data = new Float32Array(analyser.fftSize);
      const tick = () => {
        analyser.getFloatTimeDomainData(data);
        const db = _levelDb(data);
        if (typeof onLevel === "function") onLevel(db);

        // VAD: auto-stop if sustained silence after a short minimum capture
        if (vad?.enabled) {
          const now = performance.now();
          const loud = db > (vad.thresholdDb ?? -45);
          const startedMs = now - (startedAtRef.current || now);
          const minStart = vad.minStartMs ?? 250;

          if (startedMs >= minStart) {
            if (!loud) {
              if (!silenceStartRef.current) silenceStartRef.current = now;
              const silentFor = now - silenceStartRef.current;
              if (silentFor >= (vad.silenceMs ?? 1200)) {
                // stop safely
                const mr = mediaRecorderRef.current;
                if (mr && mr.state === "recording") {
                  try { mr.stop(); } catch {}
                }
              }
            } else {
              silenceStartRef.current = null;
            }
          }
        }
        rafRef.current = requestAnimationFrame(tick);
      };
      rafRef.current = requestAnimationFrame(tick);
    } catch (err) {
      console.warn("Meter/VAD init failed:", err);
    }
  }, [onLevel, vad?.enabled, vad?.silenceMs, vad?.thresholdDb, vad?.minStartMs]);

  const start = useCallback(async () => {
    if (isRecording || isProcessing || startLockRef.current) return;
    startLockRef.current = true;
    setError(null);

    if (!hasSecureMicSupport()) {
      const msg = "Mic requires HTTPS, localhost, or *.local.";
      setError(msg);
      startLockRef.current = false;
      throw new Error(msg);
    }

    // Polyfill legacy
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
          // Clean capture hints (browser may ignore some)
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: { ideal: 1 },
          sampleRate: { ideal: 16000 },
          ...(audioConstraints || {}),
          ...(deviceId ? { deviceId } : {}),
        },
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      setPermission("granted");
      streamRef.current = stream;

      // Start level meter & optional VAD
      _startMetering(stream);

      const chosen = pickSupportedMime();
      const mr = new MediaRecorder(stream, chosen ? { mimeType: chosen, audioBitsPerSecond: 128000 } : {});

      chunksRef.current = [];
      const mySession = ++sessionIdRef.current;

      mr.ondataavailable = async (e) => {
        if (sessionIdRef.current !== mySession) return; // ignore stale
        if (!e.data || e.data.size === 0) return;

        if (timesliceMs > 0 && typeof onChunk === "function") {
          // stream chunks to server
          try { await onChunk({ blob: e.data, isLast: false }); } catch (err) { console.error("onChunk error:", err); }
        } else {
          chunksRef.current.push(e.data);
        }
      };

      mr.onstop = async () => {
        if (sessionIdRef.current !== mySession) return;
        const elapsed = Math.max(0, Math.round((performance.now() - startedAtRef.current) / 1000));
        setDurationSec(elapsed);

        // build final blob if not streamed
        let finalBlob = null;
        if (timesliceMs > 0 && typeof onChunk === "function") {
          // send a tiny trailer so the receiver knows we’re done
          try { await onChunk({ blob: new Blob([], { type: mr.mimeType || "application/octet-stream" }), isLast: true }); } catch {}
        } else {
          finalBlob = new Blob(chunksRef.current, { type: mr.mimeType || "audio/webm" });
          chunksRef.current = [];
        }

        // Clear recorder ref BEFORE teardown to prevent re-entry
        mediaRecorderRef.current = null;
        setIsRecording(false);

        try {
          if (!finalBlob || !finalBlob.size) {
            // Avoid “empty reply”: send minimal beep upstream or just skip callback
            // Callers can treat missing blob as “no speech”
          } else if (typeof onBlob === "function") {
            setIsProcessing(true);
            await onBlob(finalBlob);
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
      // If timesliceMs > 0, MR will emit periodic chunks (keeps memory low)
      if (timesliceMs > 0) mr.start(timesliceMs); else mr.start();
      startedAtRef.current = performance.now();
      setIsRecording(true);

      if (maxSeconds && maxSeconds > 0) {
        timerRef.current = setTimeout(() => {
          const m = mediaRecorderRef.current;
          if (m && m.state === "recording") {
            try { m.stop(); } catch {}
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
      setTimeout(() => { startLockRef.current = false; }, 0);
    }
  }, [
    isRecording, isProcessing, maxSeconds, onBlob, onChunk,
    audioConstraints, deviceId, teardown, timesliceMs, _startMetering
  ]);

  const stop = useCallback(() => {
    const mr = mediaRecorderRef.current;
    if (!mr || mr.state !== "recording") return;
    try { mr.stop(); } catch {}
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
