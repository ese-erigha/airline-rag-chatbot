from langchain_ollama.embeddings import OllamaEmbeddings

from ..config import settings

ollamaEmbeddings = OllamaEmbeddings(
    model=settings.ollama_model,
    base_url=settings.ollama_base_url  # Adjust the base URL as per your Ollama server configuration
)
