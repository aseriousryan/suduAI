#Feel free to change the config.

# DATA_DIR_PATH = "/home/seriousco/Documents/suduAI/SUDU_AI/data" ##Change to own path
# CHUNK_SIZE = 200
# CHUNK_OVERLAP = 30
# SEPARATOR = " "
# EMBEDDER = "BAAI/bge-base-en-v1.5"
DEVICE = "cuda:0"
PROMPT_TEMPLATE = '''
With the context being provided try to answer the question. 
If you cannot answer the question based on the context either say you cannot find an answer or unable to find an answer.
So try to analyze and understand in depth about the context and answer only based on the context provided. Dont generate irrelevant answers.

<|system|>
</s>
<|user|>
Context: {context}
Question: {question}</s>
<|assistant|>


Dont make a sentence. Just provide the value. Do provide only helpful answers
'''


INP_VARS = ['context', 'question']
CHAIN_TYPE = "stuff"
# SEARCH_KWARGS = {'k': 2}
# MODEL_CKPT = "/home/seriousco/Documents/suduAI/model/zephyr-7b-beta.Q8_0.gguf" #Change to own model path
# MODEL_TYPE = "mistral"
# MAX_NEW_TOKENS = 2048


MODEL_PATH="/home/seriousco/Documents/suduAI/model/zephyr-7b-beta.Q5_K_M.gguf" #Change to own model path
TEMPERATURE = 0.5
MAX_TOKENS=2048
N_CTX=3500
TOP_P=1
GPU_LAYERS =35
VERBOSE=True

DB_PATH="chroma_db"
COLLECTION_NAME="default_collection"
FOLDER_PATH='/home/seriousco/Documents/suduAI/SUDU_AI/data'