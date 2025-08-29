from pydantic import BaseSettings

class Settings(BaseSettings):
    ENV: str = "dev"
    API_PREFIX: str = "/api"
    CORS_ORIGINS: list[str] = ["*"]
    LOG_LEVEL: str = "INFO"
    DATABASE_URL: str = "sqlite:///./english.db"
    WHISPER_MODEL: str = "base"
    PIPER_VOICE: str = "en_US-amy-medium.onnx"

    class Config:
        env_file = ".env"

settings = Settings()
