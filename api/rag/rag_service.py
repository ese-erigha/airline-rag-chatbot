from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from .chromadb_service import ChromaDatabaseService

from .embeddings import ollamaEmbeddings
from common.config import config


class RAGService:

    @staticmethod
    def query(query: str):
        chromadb_service = ChromaDatabaseService(config=config, collection_name=config.chromadb_collection,
                                                 embedding_function=ollamaEmbeddings)

        vector_store = chromadb_service.get_vector_store()
        retriever = vector_store.as_retriever(search_type="similarity")
        llm = ChatOllama(temperature=0, model=config.ollama_llm_model, base_url=config.ollama_base_url)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                               return_source_documents=True)

        query_results = qa_chain.retriever.get_relevant_documents(query)

        if not query_results:
            return "No relevant information found"

        # Combine retrieved document texts
        context = " ".join([doc.page_content for doc in query_results])

        # https://medium.com/@jyotinigam2370/customer-support-chatbot-using-rag-2934acfa9ea2
        # https://github.com/home-assistant/core/issues/121819#issuecomment-2246342069

        # augmented_prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"

        augmented_prompt = f"""
            You are an AI assistant that provides clear, structured responses **strictly using the provided knowledge base**.

            **User Question:** {query}

            **Relevant Information:**  
            {context}   
            
            **Instructions:**  
            - Answer **only** using the provided knowledge base.  
            - Provide a **step-by-step list** if applicable.  
            - Use **bullet points or numbered lists** where necessary.  
            - **Elaborate** on each step, making sure it's clear and informative. 
            - Never emphasize text. 
            - If no relevant information is found, state: 'I couldn't find enough details in my sources.'    
 

            **Final Answer:**
            """

        response = qa_chain(augmented_prompt)
        return response.get('result', "No relevant information found")
