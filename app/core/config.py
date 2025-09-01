# app/core/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    ENV: str = "dev"
    API_PREFIX: str = "/api"
    CORS_ORIGINS: List[str] = ["*"]
    LOG_LEVEL: str = "INFO"
    DATABASE_URL: str = "sqlite:///./english.db"
    WHISPER_MODEL: str = "base"
    PIPER_VOICE: str = "en_US-amy-medium.onnx"
    TTS_ENGINE: str = "bark"      # "bark" | "piper"
    BARK_DEVICE: str = "cpu"      # "cuda" or "cpu"
    BARK_VOICE: str = "v2/en_speaker_6"  # default Bark preset
    BARK_TEXT_TEMP: float = 0.7
    BARK_WAVEFORM_TEMP: float = 0.7    

    # pydantic v2 settings
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",   # ignore unknown env vars instead of raising errors
    )

settings = Settings()
