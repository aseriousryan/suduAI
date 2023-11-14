from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from config import *
from vectordb import VectorDB
from pdfloader import PDFLoader
class SuduLLM:

    def __init__(self):

        self.pdf_loader = PDFLoader()
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.prompt_temp = PROMPT_TEMPLATE
        self.input_variables = INP_VARS
        self.chain_type = CHAIN_TYPE
        self.temperature = TEMPERATURE
        self.gpu_layers = GPU_LAYERS
        self.model_path = MODEL_PATH
        self.temperature = TEMPERATURE
        self.max_tokens = MAX_TOKENS
        self.n_ctx = N_CTX
        self.top_p = TOP_P
        self.verbose = VERBOSE
    
    def create_custom_prompt(self):
        custom_prompt_temp = PromptTemplate(template=self.prompt_temp,
                            input_variables=self.input_variables)
        return custom_prompt_temp
    
    def load_llm(self):

        llm = LlamaCpp(
            model_path=self.model_path,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n_ctx=self.n_ctx,
            top_p=self.top_p,
            callback_manager=self.callback_manager,
            verbose=self.verbose, 
            n_gpu_layers=self.gpu_layers 
        )

        return llm


    def create_chain(self):
        retrieval_qa_chain = RetrievalQA.from_chain_type(
                                llm=self.llm,
                                chain_type=self.chain_type,
                                retriever=self.retriever,
                                return_source_documents=True,
                                chain_type_kwargs={"prompt": self.custom_prompt }
                            )
        return retrieval_qa_chain
    
    #initiate the llm
    def create_sudu(self):
        self.custom_prompt = self.create_custom_prompt()
        self.llm = self.load_llm()

    def get_collection(self, collection_name, db_path='default'):
        self.vector_db = VectorDB(db_path=db_path, collection_name=collection_name)
        #fileter company
        return self.vector_db.get_retriever()

    def infer_sudu(self, prompt, collection_name):
        # get collection
        self.retriever = self.get_collection(collection_name)

        # get chain
        chain = self.create_chain()

        model_out = chain(prompt)

        response = {
            'model_output': model_out['result'],
            'source_document': model_out['source_documents']
        }

        return response

if __name__ == "__main__":

    # Initialize sudu llm
    sudu_LLM = SuduLLM()

    # Create the sudu llm
    sudu_LLM.create_sudu()

