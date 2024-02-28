from FlagEmbedding import BGEM3FlagModel
from utils.common import tokenize
import os
from dotenv import load_dotenv
from utils.common import ENV

load_dotenv(f'./.env.{ENV}')

# Initialize the BGEM3FlagModel
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

def compute_embedding(df):

    # Concatenate row values into a single string
    row_string = ' '.join(map(str, df.values))

    # Row Embedding 
    row_emb = model.encode([row_string], return_dense=True)['dense_vecs']
    
    # Convert the embedded row to a list and then flatten
    flattened_emb = [item for sublist in row_emb.tolist() for item in sublist]

    return flattened_emb
