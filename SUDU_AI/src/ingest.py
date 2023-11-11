from vectordb import VectorDB
from pdfloader import PDFLoader
from typing import List

import argparse

class Ingest:
    def __init__(
        self,
        db_path: str,
        collection_name: str
    ):
        self.loader = PDFLoader()
        self.vector_db = VectorDB(db_path=db_path, collection_name=collection_name)

    def run(self, folder_path: str):
        docs = self.loader.convert(folder_path)
        doc_ids = self.vector_db.add_documents(docs)
        return doc_ids

if __name__ == '__main__':
    # Initialize Ingest with db_path and collection_name
    ingest = Ingest(db_path='db_path', collection_name='collection_name')
    # Run the ingest process with a folder_path
    doc_ids = ingest.run('folder_path')


    # ap = argparse.ArgumentParser()
    # ap.add_argument('--file_path', type=str)
    # ap.add_argument('--db_path', default='chroma_db', type=str, help='db path')
    # ap.add_argument(
    #     '--collection_name', default='default_collection', type=str, help='collection to store/retrieve data'
    # )
    # args = ap.parse_args()

    # ingest = Ingest(args.db_path, args.collection_name)
    # print(ingest.run(args.file_path))

        