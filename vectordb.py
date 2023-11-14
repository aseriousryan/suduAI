from click import progressbar

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
from typing import Dict, List
from config import *

class VectorDB:
    def __init__(
        self,
        db_path: str,
        collection_name: str,
        top_k: int=1
    ):
        self.top_k = top_k
        self.embedding_fn = HuggingFaceEmbeddings(
            model_name='BAAI/bge-large-en-v1.5',
            model_kwargs={'device': DEVICE}
        )
        
        self.db = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_fn,
            persist_directory=db_path
        )
  

    def add_documents(self, documents: List[Document]) -> List[str]:
        doc_ids = self.db.add_documents(documents=documents, progressbar=True)
        return doc_ids
    
    def get_retriever(self):
        retriever = self.db.as_retriever(search_kwargs={"k": self.top_k})
        return retriever
    
    def query(self, query: str) -> List[Document]:
        retrieved = self.db.similarity_search(query, k=self.top_k)

        return retrieved

if __name__ == '__main__':

    # Initialize VectorDB with db_path and collection_name
    vdb = VectorDB(db_path='db_path', collection_name='collection_name')


    