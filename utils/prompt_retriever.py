import sys
sys.path.append('.')

from utils.common import read_yaml, ENV
from utils.mongoDB import MongoDBController
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import os
from FlagEmbedding import BGEM3FlagModel

load_dotenv(f'./.env.{ENV}')

mongo = MongoDBController(
    host=os.environ['mongodb_url'],
    port=int(os.environ['mongodb_port']), 
    username=os.environ['mongodb_user'], 
    password=os.environ['mongodb_password']
)

model = BGEM3FlagModel(os.environ['prompt_example_retriever_sentence_transformer'],  use_fp16=True)
def prompt_example_sentence_transformer_retriever(query, database_name):
    example_desc = mongo.find_all(os.environ['mongodb_prompt_example_descriptor'], database_name)
    query_emb =  model.encode([query], return_dense=True)['dense_vecs'][0].reshape(1, -1)
    desc_emb = np.stack(example_desc['question_embedding'].values)
    cos_sims = cosine_similarity(query_emb, desc_emb)[0]
    chosen = np.argmax(cos_sims)
    question_retrieval = example_desc.iloc[chosen]['question']
    prompt_example = example_desc.iloc[chosen]['log']
    
    return prompt_example, question_retrieval


if __name__ == '_main_':
    print(prompt_example_sentence_transformer_retriever('What is the total delivery order?', 'de_carton'))