from llama_index import VectorStoreIndex, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.schema import Document

import os
import glob
import chromadb
import argparse

class VectorDB:
    def __init__(
        self,
        db: str,
        collection: str,
        top_k: int=3
    ):
        self.db = db
        self.collection = collection
        self.top_k = top_k

        self.load_storage()
        
    def load_storage(self):
        embed_model = HuggingFaceEmbedding(model_name='sentence-transformers/distiluse-base-multilingual-cased-v1')
        db = chromadb.PersistentClient(path=self.db)
        chroma_collection = db.get_or_create_collection(self.collection)
        self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.service_context = ServiceContext.from_defaults(embed_model=embed_model)

        self.index = None
        self.retrieval_engine = None

    def ingest(self, doc: Document):
        index = VectorStoreIndex.from_documents(
            documents=doc,
            storage_context=self.storage_context,
            service_context=self.service_context,
            show_progress=True
        )

    def get_retrieval_engine(self):
        if self.retrieval_engine is None:
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                service_context=self.service_context,
            )
            self.retrieval_engine = self.index.as_retriever()
        
        return self.retrieval_engine
    
    def reset_query_engine(self):
        self.index = None
        self.query_engine = None

    def retrieve(self, query: str):
        retrieval_engine = self.get_retrieval_engine()
        retrieval = retrieval_engine.retrieve(query)

        return retrieval
    

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='chroma_db', type=str, help='db path')
    ap.add_argument(
        '--collection', default='default_collection', type=str, help='collection to store/retrieve data'
    )
    args = ap.parse_args()
    
    vdb = VectorDB(db=args.db, collection=args.collection)
    retrieval = vdb.retrieve('What is the profit for 2022?')
    
    print(type(retrieval))
    print(retrieval)

