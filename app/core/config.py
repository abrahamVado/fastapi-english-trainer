from pydantic import BaseSettings

class Settings(BaseSettings):
    ENV: str = "dev"
    API_PREFIX: str = "/api"
    CORS_ORIGINS: list[str] = ["*"]
    LOG_LEVEL: str = "INFO"
    DATABASE_URL: str = "sqlite:///./english.db"
    WHISPER_MODEL: str = "base"
    PIPER_VOICE: str = "en_US-amy-medium.onnx"
    TTS_ENGINE: str = "bark"      # "bark" | "piper"
    BARK_DEVICE: str = "cpu"      # "cuda" or "cpu"
    BARK_VOICE: str = "v2/en_speaker_6"  # default Bark preset
    BARK_TEXT_TEMP: float = 0.7
    BARK_WAVEFORM_TEMP: float = 0.7    

    class Config:
        env_file = ".env"

settings = Settings()
