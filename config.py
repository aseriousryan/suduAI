#Feel free to change the config.
DEVICE = "cuda:0"
PROMPT_TEMPLATE = '''

With the context being provided try to answer the question. 
If you cannot answer the question based on the context either say you cannot find an answer or unable to find an answer.
So try to analyze and understand in depth about the context and answer only based on the context provided. Dont generate irrelevant answers.

<|system|>
You are a professional and experience sales financial analyst, you are good in sales reports analysis and summarizations. 
Answer the user's query based on the context given in short, professional and precise.
<|user|>
Context: {context}
Question: {question}</s>
<|assistant|>

Dont make a sentence. Just provide the value. Do provide only helpful answers and arrange the answers nicely in bullet forms.
Make sure the output texts are in same font style.
'''
INP_VARS = ['context', 'question']
CHAIN_TYPE = "stuff"
MODEL_PATH="/home/seriousco/Documents/suduAI/model/zephyr-7b-beta.Q5_K_M.gguf" #Change to own model path
TEMPERATURE = 0.5
MAX_TOKENS=3000
N_CTX=6000
TOP_P=1
GPU_LAYERS =35
VERBOSE=True
DB_PATH="chroma_db"
COLLECTION_NAME="default_collection"
FOLDER_PATH='/home/seriousco/Documents/suduAI/SUDU_AI/data' #Folder for input data