from utils.mongoDB import MongoDBController
from utils.common import tokenize, read_yaml
from dotenv import load_dotenv

import pandas as pd

import os
import io

load_dotenv('./.env')

mongo = MongoDBController(
    host=os.environ['mongodb_url'],
    port=int(os.environ['mongodb_port']), 
    username=os.environ['mongodb_user'], 
    password=os.environ['mongodb_password']
)

def get_table_description(df, desc=''):
    df_head = df.head(5).to_markdown()
    string_buffer = io.StringIO()
    df.info(buf=string_buffer)
    df_info = string_buffer.getvalue()

    categorical_columns = df.select_dtypes(include=['object', 'category'])

    categorical_desc = ''
    for col in categorical_columns:
        unique_values = df[col].unique().tolist()
        token_length = [len(tokenize(os.environ['tokenizer'], value)) for value in unique_values]
        
        # if sum of token length is more than 500, don't include column into description
        if sum(token_length) >= 500: continue
        categorical_desc += f'{col}: {unique_values}\n'

    final_desc = f'This is the quantitative information of the table:\n{df_info}\n'
    if len(categorical_desc) > 0:
        final_desc += f'The following is a list of categorical columns and their possible values:\n{categorical_desc}\n'
    if len(desc) > 0:
        final_desc = f'{desc}\n{final_desc}'

    desc_token_length = len(tokenize(os.environ['tokenizer'], final_desc))

    return final_desc, desc_token_length
    