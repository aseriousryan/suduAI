from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.chains import LLMChain


import uvicorn
import os
import traceback



app = FastAPI()

@app.post('/ingest')
async def ingest():
    try:
        #Own Refs
        # template = """[INST] <<SYS>>
        # You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  
        # Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
        # Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, 
        # or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, 
        # please don't share false information.
        # <</SYS>>
        # {question}[/INST]
        # """

        # prompt = PromptTemplate(template=template, input_variables=["question"])
       
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


        llm = LlamaCpp(
            model_path="model/yarn-mistral-7b-128k.Q8_0.gguf", #change path accordingly
            temperature=0.2,
            top_p=1,
            n_ctx=6000,
            callback_manager=callback_manager, 
            verbose=True,
            n_gpu_layers=30
        )
        
        loader = PyPDFLoader("input_data/llama2.pdf") #Will change to accept multiple files soon in a folder.
        documents = loader.load()

        # quick check on the loaded document for the correct pages etc
        print(len(documents), documents[0].page_content[0:300])

        # split the loaded documents into chunks 
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20, separators=["\n\n", "\n", " ", ""])
        
        all_splits = text_splitter.split_documents(documents)

        persist_directory = 'data'

        # create the vector db to store all the split chunks as embeddings
        embeddings = HuggingFaceEmbeddings()
        vectordb = Chroma.from_documents(
            all_splits,
            embeddings,
            persist_directory=persist_directory
        )

        vectordb.persist()        
        print(vectordb._collection.count())
    
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectordb.as_retriever(),
            chain_type="refine"
        )


        question = "What is Llama 2 and its function?"
        docs = vectordb.similarity_search(question,k=3)
        len(docs)
        result = qa_chain({"query": question})
        print(result["result"])

    except:
        return HTTPException(status_code=404, detail=traceback.format_exc())

    return True

if __name__ == '__main__':
    uvicorn.run('main:app', host="0.0.0.0", port=8085, reload=True)















