from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ollama_base_url: str
    ollama_generate_path: str
    ollama_model: str
    chromadb_host: str
    chromadb_port: int
    chromadb_collection: str


settings = Settings()
