# app/tools/freeze_bark_voice.py
from pathlib import Path
from bark.generation import save_as_prompt

VOICE_PATH = Path(__file__).resolve().parents[1] / "voices" / "es_0.npz"
VOICE_PATH.parent.mkdir(parents=True, exist_ok=True)

save_as_prompt(str(VOICE_PATH), history_prompt="v2/es_speaker_0")
print(f"[bark] saved preset to: {VOICE_PATH}")
