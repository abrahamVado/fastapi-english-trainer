// src/hooks/useRecorder.js
import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Candidate MIME types to try in order of preference.
 * Most modern desktop browsers support "audio/webm;codecs=opus".
 * Safari may not support webm; if none of these are supported, we let MediaRecorder pick.
 */
const MIME_CANDIDATES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/ogg;codecs=opus",
  "audio/ogg",
];

/** Pick the first MediaRecorder mimeType the browser claims to support. */
function pickSupportedMime() {
  for (const m of MIME_CANDIDATES) {
    if (window.MediaRecorder?.isTypeSupported?.(m)) return m;
  }
  return "";
}

/** True if this origin can use getUserMedia (HTTPS or http://localhost). */
function hasSecureMicSupport() {
  if (window.isSecureContext) return true;
  // allow localhost (browsers treat it as a secure context)
  return /^(localhost|127\.0\.0\.1)$/.test(location.hostname);
}

/**
 * useRecorder — ergonomic mic recorder hook using MediaRecorder.
 *
 * Features:
 *  - Secure-context & capability checks with friendly errors (works over HTTPS or localhost).
 *  - Optional hotkeys: Space (hold-to-record), Esc (cancel).
 *  - onBlob can be async; hook shows isProcessing during your upload/transcribe.
 *  - Permission state reflection (unknown/granted/denied).
 *  - Optional advanced constraints/deviceId.
 *
 * @param {Object} opts
 * @param {number}  [opts.maxSeconds=60]          Auto-stop after this many seconds (set null/0 to disable).
 * @param {function} opts.onBlob                  Callback(blob). Can return a Promise.
 * @param {boolean} [opts.enableHotkeys=true]     Space/Esc bindings.
 * @param {boolean} [opts.autoStopOnBlur=true]    Stop recording if page becomes hidden.
 * @param {MediaTrackConstraints} [opts.audioConstraints] Extra getUserMedia audio constraints.
 * @param {string} [opts.deviceId]                Specific microphone device id to use.
 */
export function useRecorder({
  maxSeconds = 60,
  onBlob,
  enableHotkeys = true,
  autoStopOnBlur = true,
  audioConstraints,
  deviceId,
} = {}) {
  // --- Public state ---------------------------------------------------------
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [mimeType, setMimeType] = useState("");            // chosen mimeType
  const [durationSec, setDurationSec] = useState(0);       // last recording duration (approx)
  const [permission, setPermission] = useState("unknown"); // "unknown" | "granted" | "denied"

  // --- Internals ------------------------------------------------------------
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);
  const startedAtRef = useRef(0);

  /** Best-effort polyfill to reflect current microphone permission (if supported). */
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        if (navigator.permissions?.query) {
          const st = await navigator.permissions.query({ name: "microphone" });
          if (!cancelled) {
            setPermission(st.state); // "granted" | "prompt" | "denied" (map "prompt" to "unknown")
            st.onchange = () => setPermission(st.state);
          }
        }
      } catch {
        /* ignore */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  /** Teardown helper: stops recorder + mic tracks, clears timer. */
  const teardown = useCallback(() => {
    try {
      mediaRecorderRef.current?.stop?.();
    } catch {}
    mediaRecorderRef.current = null;

    try {
      streamRef.current?.getTracks?.().forEach((t) => t.stop());
    } catch {}
    streamRef.current = null;

    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  /**
   * Start recording:
   *  - Checks secure context & feature availability (clear message if LAN over http).
   *  - Requests mic with optional constraints/deviceId.
   *  - Picks best mimeType and wires MediaRecorder.
   *  - Auto-stops at maxSeconds if configured.
   */
  const start = useCallback(async () => {
    if (isRecording || isProcessing) return;
    setError(null);

    // 1) Security/capability guards up-front
    if (!hasSecureMicSupport()) {
      const msg =
        "Mic requires HTTPS or http://localhost. Open the app at https://<your-ip>:5173 or use localhost.";
      setError(msg);
      throw new Error(msg);
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      // Try to gently polyfill from legacy APIs (very old browsers)
      const legacyGetUserMedia =
        navigator.getUserMedia ||
        navigator.webkitGetUserMedia ||
        navigator.mozGetUserMedia;
      if (legacyGetUserMedia) {
        navigator.mediaDevices = navigator.mediaDevices || {};
        navigator.mediaDevices.getUserMedia = (c) =>
          new Promise((res, rej) => legacyGetUserMedia.call(navigator, c, res, rej));
      } else {
        const msg =
          "getUserMedia is not available in this context. Use a modern browser over HTTPS or localhost.";
        setError(msg);
        throw new Error(msg);
      }
    }

    try {
      // 2) Build constraints
      const constraints = {
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          ...(audioConstraints || {}),
          ...(deviceId ? { deviceId } : {}),
        },
      };

      // 3) Ask for permission (prompts once)
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      setPermission("granted");
      streamRef.current = stream;

      // 4) Choose MIME
      const chosen = pickSupportedMime();
      setMimeType(chosen || "(browser default)");

      // 5) Create recorder
      const mr = new MediaRecorder(stream, chosen ? { mimeType: chosen } : undefined);
      chunksRef.current = [];

      mr.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
      };

      // When recording stops → build blob, hand off to onBlob (await it if async)
      mr.onstop = async () => {
        const elapsed = Math.max(
          0,
          Math.round((performance.now() - startedAtRef.current) / 1000)
        );
        setDurationSec(elapsed);

        const blob = new Blob(chunksRef.current, {
          type: mr.mimeType || "audio/webm",
        });
        chunksRef.current = [];

        if (blob.size && typeof onBlob === "function") {
          try {
            setIsProcessing(true);
            await onBlob(blob); // support async
          } catch (err) {
            console.error("onBlob error:", err);
            setError(err?.message || String(err));
          } finally {
            setIsProcessing(false);
          }
        } else {
          setIsProcessing(false);
        }

        teardown();
      };

      // 6) Kick off recording
      mediaRecorderRef.current = mr;
      mr.start();
      startedAtRef.current = performance.now();
      setIsRecording(true);

      // 7) Optional safety: auto-stop after maxSeconds
      if (maxSeconds && maxSeconds > 0) {
        timerRef.current = setTimeout(() => {
          stop(); // graceful stop
        }, maxSeconds * 1000);
      }
    } catch (err) {
      console.error(err);
      setError(err?.message || String(err));
      setPermission(err?.name === "NotAllowedError" ? "denied" : "unknown");
      setIsRecording(false);
      setIsProcessing(false);
      teardown();
      throw err; // surface to caller if needed
    }
  }, [isRecording, isProcessing, maxSeconds, onBlob, audioConstraints, deviceId, teardown]);

  /** Stop recording and finalize the blob (invokes onBlob). */
  const stop = useCallback(() => {
    const mr = mediaRecorderRef.current;
    if (!mr || mr.state !== "recording") return;
    try {
      mr.stop(); // triggers onstop handler above
    } catch {}
    setIsRecording(false);
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  /** Cancel recording: discard chunks and stop without emitting a blob. */
  const cancel = useCallback(() => {
    const mr = mediaRecorderRef.current;
    if (!mr) return;
    chunksRef.current = [];
    try {
      if (mr.state === "recording") mr.stop();
    } catch {}
    setIsRecording(false);
    setIsProcessing(false);
    teardown();
  }, [teardown]);

  // Optional: hotkeys — Space = press & hold to record, Esc = cancel
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

  // Optional: auto-stop if the tab becomes hidden (prevents background recording issues)
  useEffect(() => {
    if (!autoStopOnBlur) return;
    const onVis = () => {
      if (document.hidden && isRecording) stop();
    };
    document.addEventListener("visibilitychange", onVis);
    return () => document.removeEventListener("visibilitychange", onVis);
  }, [autoStopOnBlur, isRecording, stop]);

  // Cleanup on unmount (stop tracks/clear timers)
  useEffect(() => teardown, [teardown]);

  return {
    // states
    isRecording,
    isProcessing,
    error,
    mimeType,
    durationSec,
    permission, // "unknown" | "granted" | "denied"

    // controls
    start,
    stop,
    cancel,

    // advanced: let callers force/reflect UI state if they need
    setIsProcessing,
  };
}
