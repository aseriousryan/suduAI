import sys
sys.path.append('.')

from utils.common import read_yaml, ENV, generate_sql_table_schema_markdown
from utils.mongoDB import MongoDBController
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

import os

load_dotenv(f'./.env.{ENV}')

mongo = MongoDBController(
    host=os.environ['mongodb_url'],
    port=int(os.environ['mongodb_port']), 
    username=os.environ['mongodb_user'], 
    password=os.environ['mongodb_password']
)

model = SentenceTransformer(os.environ['collection_retriever_sentence_transformer'], device='cuda')

def table_schema_retriever(query, database_name):
    df_schema = mongo.find_all(os.environ['mongodb_sql_table_schema'], database_name)
    query_emb = model.encode(query, convert_to_numpy=True).reshape(1, -1)

    desc_emb = np.stack(df_schema['embedding'].values)
    cos_sims = cosine_similarity(query_emb, desc_emb)[0]
    chosen = np.argmax(cos_sims)
    table_name = df_schema.iloc[chosen]['table_name']
    retrieval_description = df_schema.iloc[chosen]['retrieval_description']
    column_name = df_schema.iloc[chosen]['column_name']
    data_type = df_schema.iloc[chosen]['data_type']
    
    table_schema_markdown = generate_sql_table_schema_markdown(table_name, column_name, data_type)
    
    return table_name, table_schema_markdown, retrieval_description, cos_sims[chosen]


if __name__ == '__main__':
    print(table_schema_retriever('What is the total delivery order?', 'de_carton'))