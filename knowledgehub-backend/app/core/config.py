from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "nvidia"
    nvidia_api_key: str = ""
    nvidia_llm_model: str = "meta/llama-3.1-8b-instruct"
    nvidia_embedding_model: str = "nvidia/llama-nemotron-embed-1b-v2"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "knowledgehub_documents"

    database_url: str = "sqlite:///./knowledgehub.db"

    max_upload_size_mb: int = 20
    upload_dir: str = "./uploads"

    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.25
    chat_history_turns: int = 6

    cors_origins: str = "http://localhost:3000,http://localhost:3005"


settings = Settings()
