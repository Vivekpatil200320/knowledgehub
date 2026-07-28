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

    retrieval_top_k: int = 6
    # Refusal is judged on the BEST hit only: if nothing in the corpus is even close,
    # the question isn't answerable here. Measured separation on the eval corpus is
    # wide — in-corpus questions top out around 0.45-0.50, out-of-corpus below 0.12.
    refusal_score_threshold: float = 0.20
    # Supporting chunks only need to be plausibly related to earn a place in the
    # context window. Kept well below the refusal bar so that a question which clearly
    # IS answerable still gets its lower-ranked evidence (e.g. the education section
    # of a resume, which scores far below the header block).
    context_score_floor: float = 0.05
    # What the model READS and what gets shown as a "Source" citation chip are
    # different questions. A cross-document noise chunk clears context_score_floor
    # easily enough to be read, but showing it as a source is what erodes user trust —
    # "why does this pricing answer cite my resume?". Measured on live corpus queries,
    # a correct document's own supporting chunks score 0.68-0.94 of that query's top
    # hit; unrelated documents' noise scores 0.20-0.40. 0.5 sits in the middle of that
    # gap with margin on both sides.
    citation_relative_floor: float = 0.5
    # A relative floor alone has a failure mode that shows up exactly where it hurts
    # most: the cutoff is a fraction of the top hit, so a weak top hit produces a weak
    # cutoff. Measured — a question whose best match scored 0.36 ("what is Acme's SOC 2
    # renewal date?", genuinely absent from the corpus) dropped the bar to 0.180 and
    # admitted unrelated telemedicine chunks at 0.19, so an answer that correctly said
    # "that isn't in the documents" still displayed two confident, irrelevant sources.
    # The relative floor got loosest at the precise moment the answer was least certain.
    # This absolute floor sits below every legitimate supporting citation observed
    # (0.31-0.52) and above cross-document noise (0.19-0.20); both bars must be cleared.
    citation_absolute_floor: float = 0.25
    chat_history_turns: int = 6

    cors_origins: str = "http://localhost:3000,http://localhost:3005"


settings = Settings()
