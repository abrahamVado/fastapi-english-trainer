import { useCallback, useEffect, useRef, useState } from "react";

const mimeOptions = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/ogg;codecs=opus",
  "audio/ogg",
];

function pickSupportedMime() {
  for (const m of mimeOptions) {
    if (window.MediaRecorder?.isTypeSupported?.(m)) return m;
  }
  return "";
}

export function useRecorder({ maxSeconds = 60, onBlob }) {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  const start = useCallback(async () => {
    if (isRecording || isProcessing) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = pickSupportedMime();
      const mr = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);

      chunksRef.current = [];
      mr.ondataavailable = (e) => e.data?.size && chunksRef.current.push(e.data);
      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mr.mimeType || "audio/webm" });
        chunksRef.current = [];
        if (blob.size) onBlob?.(blob);
        setIsProcessing(true);
      };

      mediaRecorderRef.current = mr;
      mr.start();
      setIsRecording(true);

      setTimeout(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") stop();
      }, maxSeconds * 1000);
    } catch (err) {
      console.error(err);
      setIsRecording(false);
      setIsProcessing(false);
      throw err;
    }
  }, [isRecording, isProcessing, maxSeconds, onBlob]);

  const stop = useCallback((cancel = false) => {
    const mr = mediaRecorderRef.current;
    if (!mr || mr.state !== "recording") return;

    if (cancel) {
      chunksRef.current = [];
      setIsProcessing(false);
    }
    mr.stop();
    setIsRecording(false);
  }, []);

  useEffect(() => {
    const down = (e) => {
      if (e.code === "Space" && !e.repeat) {
        e.preventDefault();
        if (!isRecording && !isProcessing) start();
      }
      if (e.code === "Escape") {
        e.preventDefault();
        if (isRecording) stop(true);
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
  }, [isRecording, isProcessing, start, stop]);

  return { isRecording, isProcessing, setIsProcessing, start, stop };
}
