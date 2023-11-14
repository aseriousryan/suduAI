from langchain.document_loaders import UnstructuredFileLoader, PyPDFLoader
from langchain.schema import Document
from typing import List
from langchain.document_loaders import DirectoryLoader

class PDFLoader:

    def __init__(self):
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

