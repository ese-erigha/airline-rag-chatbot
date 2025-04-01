from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from .chromadb_service import ChromaDatabaseService

from .embeddings import ollamaEmbeddings
from ..config import settings


class RAGService:

    @staticmethod
    def query(prompt: str, query: str):
        chromadb_service = ChromaDatabaseService(collection_name=settings.chromadb_collection,
                                                 embedding_function=ollamaEmbeddings)

        vector_store = chromadb_service.get_vector_store()
        retriever = vector_store.as_retriever(search_type="similarity")
        llm = ChatOllama(temperature=0, model=settings.ollama_model, base_url=settings.ollama_base_url)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                               return_source_documents=True)

        query_results = qa_chain.retriever.get_relevant_documents(query)
        if not query_results:
            return "No relevant information found"

        # https://medium.com/@jyotinigam2370/customer-support-chatbot-using-rag-2934acfa9ea2
        response = qa_chain({"query": query})
        return response.get('result', "No response generated")
