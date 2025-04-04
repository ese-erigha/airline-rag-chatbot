from fastapi import FastAPI

from rag.data_loader_service import DataLoaderService
from rag.indexer_service import IndexerService
from rag.rag_service import RAGService
from rag.chromadb_service import ChromaDatabaseService
from pydantic import BaseModel
from common.config import config
from rag.embeddings import ollamaEmbeddings


class QueryInput(BaseModel):
    query: str


app = FastAPI()


@app.get('/')
def home():
    return {"Chat": "Bot"}


@app.get('/index/build')
def build_index():
    docs = DataLoaderService.load_pdf("./rag/airlines.pdf")
    IndexerService.index_documents(documents=docs)
    return {"success": True}


@app.post('/rag/query')
def rag_query(query_input: QueryInput) -> dict[str, str]:
    response: str = RAGService.query(query=query_input.query)
    return {"summary": response}


@app.get('/index/documents')
def rag_query():
    db_service = ChromaDatabaseService(config=config, collection_name=config.chromadb_collection,
                                       embedding_function=ollamaEmbeddings)

    result = db_service.get_documents(collection_name=config.chromadb_collection)
    print(result)
    return {"documents": result["documents"]}