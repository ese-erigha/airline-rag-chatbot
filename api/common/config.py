from pydantic_settings import BaseSettings


class Config(BaseSettings):
    ollama_base_url: str
    ollama_generate_path: str
    ollama_llm_model: str
    ollama_embedding_model: str
    chromadb_host: str
    chromadb_port: int
    chromadb_collection: str


config = Config()
