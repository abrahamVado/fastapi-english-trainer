// src/lib/audio.js
// Play a Blob (WAV/OGG/etc) in the browser.

export async function playBlob(blob) {
  const url = URL.createObjectURL(blob);
  try {
    const audio = new Audio(url);
    audio.onended = () => URL.revokeObjectURL(url);
    await audio.play();
  } finally {
    // in case play() throws (autoplay policy), cleanup after a while
    setTimeout(() => URL.revokeObjectURL(url), 30_000);
  }
}
