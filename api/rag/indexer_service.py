from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from ..config import settings
from .chromadb_service import ChromaDatabaseService
from .embeddings import ollamaEmbeddings


class IndexerService:

    @staticmethod
    def split_documents(docs: list[Document]) -> list[Document]:
        separators = ["\n\n", "\n", ".", ""]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100,
                                                       separators=separators)
        return text_splitter.split_documents(docs)

    @staticmethod
    def index_documents(self, documents: list[Document]):
        document_splits = IndexerService.split_documents(documents)

        ids = [str(idx) for idx, _ in enumerate(document_splits)]

        # Create ChromaDB service
        chromadb_service = ChromaDatabaseService(collection_name=settings.chromadb_collection,
                                                 embedding_function=ollamaEmbeddings)

        chromadb_service.add_documents_to_collection(documents=document_splits, ids=ids)

    # @staticmethod
    # def index_documents(self, documents: list[Document]):
    #     document_splits = IndexerService.split_documents(documents)
    #
    #     # Create ollama embeddings function
    #     embedding_function = OllamaEmbeddings(
    #         model=settings.ollama_model,
    #         base_url=settings.ollama_base_url  # Adjust the base URL as per your Ollama server configuration
    #     )
    #
    #     docs = [doc.page_content for doc in document_splits]
    #     ids = [str(idx) for idx, _ in enumerate(document_splits)]
    #
    #     # Create ChromaDB service
    #     chromadb_service = ChromaDatabaseService()
    #     chromadb_service.add_documents_to_collection(docs=docs, ids=ids, collection_name=settings.chromadb_collection,
    #                                                  embedding_function=embedding_function)


