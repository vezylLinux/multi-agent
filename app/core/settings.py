from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_env: str = "dev"
    frontend_origin: str = "http://localhost:3000"
    chroma_mode: str = "persistent"
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_persist_dir: str = "data/chroma"
    chroma_collection_name: str = "travel_place_chunks"
    rag_chunk_size_words: int = 120
    rag_chunk_overlap_words: int = 24
    rag_context_chunks: int = 6
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 32
    google_maps_api_key: str = ""
    google_places_base_url: str = "https://places.googleapis.com"
    google_places_enrich_enabled: bool = False
    google_places_override_coordinates: bool = False
    google_places_language_code: str = "vi"
    google_places_region_code: str = "VN"
    google_places_request_timeout_s: int = 10
    google_places_follow_moved_limit: int = 2
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openai/gpt-oss-120b:free"
    openrouter_reasoning_enabled: bool = True
    openrouter_request_timeout_s: int = 25
    trackasia_api_key: str = ""
    trackasia_enabled: bool = True
    trackasia_geocode_enabled: bool = True
    trackasia_routing_enabled: bool = True
    trackasia_directions_base_url: str = "https://maps.track-asia.com/route/v2/directions"
    trackasia_request_timeout_s: int = 8
    trackasia_new_admin: bool = True
    trackasia_cache_ttl_s: int = 900
    trackasia_rate_limit_window_s: int = 60
    trackasia_rate_limit_max_calls: int = 60
    trackasia_route_modes: str = "car"
    database_url: str = "sqlite:///data/travel.db"
    session_cookie_name: str = "app_session"
    session_cookie_secure: bool = False
    session_cookie_samesite: str = "lax"
    session_cookie_max_age_seconds: int = 60 * 60 * 24 * 30

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
