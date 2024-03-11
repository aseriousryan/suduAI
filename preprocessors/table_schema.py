from utils.mongoDB import MongoDBController
from utils.common import tokenize, ENV
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

import pandas as pd

import os
import io

load_dotenv(f'./.env.{ENV}')

emb_model = SentenceTransformer(os.environ['collection_retriever_sentence_transformer'])

def get_table_schema(df,retrieval_desc=None):
    # Function to map pandas data types to string data types
    def map_data_type(dtype):
        if pd.api.types.is_integer_dtype(dtype):
            return 'int'
        elif pd.api.types.is_float_dtype(dtype):
            return 'decimal'
        elif pd.api.types.is_string_dtype(dtype):
            return 'varchar'
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return 'datetime'
        else:
            return 'unknown'

    # Convert data types to string representations
    data_types = df.dtypes.map(map_data_type).astype(str).tolist()

    # Combine column names and data types in the desired format
    columns_and_types = [f'{col}; {dtype}' for col, dtype in zip(df.columns, data_types)]

    # Convert the list to strings
    columns_str = '; '.join(df.columns)
    data_types_str = '; '.join(data_types)
    
    emb = []
    if retrieval_desc is not None:
        emb = emb_model.encode(retrieval_desc, convert_to_numpy=True).tolist()

    desc_token_length = len(tokenize(os.environ['tokenizer'], retrieval_desc))

    return columns_str, data_types_str, retrieval_desc, desc_token_length, emb
    