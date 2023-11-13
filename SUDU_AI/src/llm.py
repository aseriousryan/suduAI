from langchain import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from config import *
from vectordb import VectorDB
from pdfloader import PDFLoader

class SuduBotCreator:

    def __init__(self):
        # Initialize PDFLoader, Ingest, and VectorDB
        self.pdf_loader = PDFLoader()

        # Other initializations...
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
    
    def create_sudu_bot(self):
        self.custom_prompt = self.create_custom_prompt()
        self.llm = self.load_llm()

    def get_collection(self, collection_name, db_path='default'):
        self.vector_db = VectorDB(db_path=db_path, collection_name=collection_name)
        return self.vector_db.get_retriever()

    def infer_sudu_bot(self, prompt, collection_name):
        # get collection
        self.retriever = self.get_collection(collection_name)
        # get chain
        chain = self.create_chain()

        model_out = chain(prompt)

        response = {
            'model_output': model_out,
            'source_document': model_out['source_documents']
        }

        return response

        # answer = model_out['result']
        # source_documents = model_out['source_documents']
        # metadata = [doc.metadata for doc in source_documents]

        # # Create a structured string for the output
        # output = f"{answer}"
        # for i, meta in enumerate(metadata):
        #     output += f"\nSource Document {i+1}:\n"
        #     output += f"Page: {meta['page']}\n"
        #     output += f"Source: {meta['source']}\n"

        # return output

if __name__ == "__main__":
    # Initialize SuduBotCreator with the parsed arguments
    sudu_bot_creator = SuduBotCreator()

    # Create the bot
    sudu_bot_creator.create_sudu_bot()

    # Now you can call the infer_sudu_bot method
    output = sudu_bot_creator.infer_sudu_bot({'query': 'Who are the investors press person?'}, 'collection_name')
    print(output)


