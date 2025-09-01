// src/lib/recorder.js
// Simple one-shot recorder using MediaRecorder → resolves with a Blob (audio/webm)

export async function recordOnce({ maxMs = 8000 } = {}) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const rec = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });

  const chunks = [];
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      if (rec.state !== "inactive") rec.stop();
    }, maxMs);

    rec.ondataavailable = (e) => e.data && e.data.size && chunks.push(e.data);
    rec.onerror = (e) => {
      clearTimeout(timer);
      stream.getTracks().forEach((t) => t.stop());
      reject(e);
    };
    rec.onstop = () => {
      clearTimeout(timer);
      stream.getTracks().forEach((t) => t.stop());
      resolve(new Blob(chunks, { type: "audio/webm" }));
    };

    rec.start();
  });
}
