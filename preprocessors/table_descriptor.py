from utils.mongoDB import MongoDBController
from utils.common import tokenize, ENV
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

import pandas as pd

import os
import io

load_dotenv(f'./.env.{ENV}')

emb_model = SentenceTransformer(os.environ['collection_retriever_sentence_transformer'])

def get_table_description(df, desc='', retrieval_desc=None):
    df_head = df.head(5).to_markdown()
    string_buffer = io.StringIO()
    df.info(buf=string_buffer)
    df_info = string_buffer.getvalue()

    categorical_columns = df.select_dtypes(include=['object', 'category'])

    categorical_desc = ''
    for col in categorical_columns:
        unique_values = df[col].unique().tolist()
        token_length = [len(tokenize(os.environ['tokenizer'], str(value))) for value in unique_values]
        
        # if sum of token length is more than 200, don't include column into description
        if sum(token_length) >= 200: continue
        categorical_desc += f'{col}: {unique_values}\n'

    final_desc = f'This is the quantitative information of the table:\n{df_info}\n'
    if len(categorical_desc) > 0:
        final_desc += f'The following is a list of categorical columns and their possible values:\n{categorical_desc}\n'
    
    emb = []
    if retrieval_desc is not None:
        emb = emb_model.encode(desc, convert_to_numpy=True).tolist()
    elif len(desc) > 0:
        emb = emb_model.encode(desc, convert_to_numpy=True).tolist()

    if len(desc) > 0:
        final_desc = f'{desc}\n{final_desc}'

    desc_token_length = len(tokenize(os.environ['tokenizer'], final_desc))

    return final_desc, desc_token_length, emb
    