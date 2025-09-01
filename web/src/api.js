export const API = {
  audio: import.meta.env.VITE_API_AUDIO || "/api/audio",
  tts:   import.meta.env.VITE_API_TTS   || "/api/tts",
  ttsWarm: import.meta.env.VITE_API_TTS_WARM || "/api/tts/warm",
};
