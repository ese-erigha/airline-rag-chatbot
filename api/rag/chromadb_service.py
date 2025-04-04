import chromadb
# from chromadb import Documents, EmbeddingFunction
from chromadb.api import ClientAPI
from chromadb.api.types import IncludeEnum

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from common.config import Config


# https://python.langchain.com/docs/integrations/vectorstores/chroma/

# Define a custom embedding function for ChromaDB using Ollama
# class ChromaDBEmbeddingFunction(EmbeddingFunction):
#     """
#     Custom embedding function for ChromaDB using embeddings from Ollama.
#     """
#
#     def __init__(self, langchain_embeddings):
#         self.langchain_embeddings = langchain_embeddings
#         super().__init__()
#
#     def __call__(self, items: Documents) -> Embeddings:
#         docs = items
#         # Ensure the input is in a list format for processing
#         if isinstance(docs, str):
#             docs = [docs]
#         return self.langchain_embeddings.embed_documents(docs)


class ChromaDatabaseService:
    client: ClientAPI
    vector_store: Chroma

    def __init__(self, config: Config, collection_name: str, embedding_function: Embeddings):
        self.client = chromadb.HttpClient(host=config.chromadb_host, port=config.chromadb_port)
        self.vector_store = Chroma(collection_name=collection_name, embedding_function=embedding_function,
                                   client=self.client)

    def get_vector_store(self):
        return self.vector_store

    def get_documents(self, collection_name: str):
        collection = self.client.get_or_create_collection(name=collection_name)
        return collection.get(include=[IncludeEnum.documents])

    def add_documents_to_collection(self, documents: list[Document], ids: list[str]):
        self.vector_store.add_documents(ids=ids, documents=documents)

    # def add_documents_to_collection(self, docs: list[str], ids: list[str], collection_name: str,
    #                                 embedding_function: Embeddings):
    #     # Initialize the embedding function with Ollama embeddings
    #     embedding = ChromaDBEmbeddingFunction(
    #         embedding_function
    #     )
    #     collection = self.client.get_or_create_collection(name=collection_name, embedding_function=embedding)
    #     collection.add(documents=docs, ids=ids)
    #
    # vector_store = Chroma(collection_name=collection_name, embedding_function=embedding_function,
    # client=self.client) vector_store.add_documents(ids=ids)
