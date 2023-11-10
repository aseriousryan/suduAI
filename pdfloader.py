from langchain.document_loaders import UnstructuredFileLoader, PyPDFLoader
from langchain.schema import Document
from typing import List

import argparse

class PDFLoader:
    def __init__(self):
        self.loader = PyPDFLoader

    def convert(self, file_path: str) -> List[Document]:
        data = self.loader(file_path).load()

        return data
    
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--file_path', type=str)
    args = ap.parse_args()

    loader = PDFLoader()
    docs = loader.convert(args.file_path)

    for doc in docs:
        print(f'page: {doc.page_content}')
        print(f'metadata: {doc.metadata}')
        break
