from langchain.document_loaders import UnstructuredFileLoader, PyPDFLoader
from langchain.schema import Document
from typing import List
from config import *
import argparse
from langchain.document_loaders import DirectoryLoader

class PDFLoader:
    def __init__(self):
        # self.loader = PyPDFLoader
        
        self.loader = DirectoryLoader

    def convert(self, folder_path: str) -> List[Document]:
        data = self.loader(folder_path,
                            glob='*.pdf',
                            loader_cls=PyPDFLoader).load()

        return data
    
if __name__ == '__main__':
    # Initialize PDFLoader
    loader = PDFLoader()
    # Convert a PDF file to a list of Documents
    docs = loader.convert('folder_path')


    # ap = argparse.ArgumentParser()
    # ap.add_argument('--file_path', type=str)
    # args = ap.parse_args()

    # loader = PDFLoader()
    # docs = loader.convert(args.file_path)

    # for doc in docs:
    #     print(f'page: {doc.page_content}')
    #     print(f'metadata: {doc.metadata}')
    #     break
