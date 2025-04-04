from langchain_ollama.embeddings import OllamaEmbeddings

from common.config import config

ollamaEmbeddings = OllamaEmbeddings(
    model=config.ollama_embedding_model,
    base_url=config.ollama_base_url  # Adjust the base URL as per your Ollama server configuration
)
