#Feel free to change the config.
DATA_DIR_PATH = "/home/seriousco/Documents/suduAI/NLP-Projects-NHV/LLMs Related/Zephyr Research to Production/RAG_Invoice_Processing/data" ##Change to own path
CHUNK_SIZE = 200
CHUNK_OVERLAP = 30
SEPARATOR = " "
EMBEDDER = "BAAI/bge-base-en-v1.5"
DEVICE = "cuda:0"
PROMPT_TEMPLATE = '''
With the information being provided try to answer the question. 
If you cant answer the question based on the information either say you cant find an answer or unable to find an answer.
So try to understand in depth about the context and answer only based on the information provided. Dont generate irrelevant answers

Context: {context}
Question: {question}
Dont make a sentence. Just provide the value. Do provide only helpful answers

Helpful answer:
'''
INP_VARS = ['context', 'question']
CHAIN_TYPE = "stuff"
SEARCH_KWARGS = {'k': 2}
MODEL_CKPT = "/home/seriousco/Documents/suduAI/model/yarn-mistral-7b-128k.Q8_0.gguf" #Change to own model path
MODEL_TYPE = "mistral"
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.0