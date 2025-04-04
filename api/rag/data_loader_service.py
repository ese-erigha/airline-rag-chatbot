import os
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_core.documents import Document


class DataLoaderService:
    @staticmethod
    def load_pdf(file_path: str) -> list[Document]:
        loader = PyPDFium2Loader(file_path)
        documents = loader.load()
        return documents
