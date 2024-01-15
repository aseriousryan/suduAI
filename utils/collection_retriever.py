import sys
sys.path.append('.')

from utils.common import read_yaml, ENV
from utils.mongoDB import MongoDBController
from dotenv import load_dotenv

import os

load_dotenv(f'./.env.{ENV}')

mongo = MongoDBController(
    host=os.environ['mongodb_url'],
    port=int(os.environ['mongodb_port']), 
    username=os.environ['mongodb_user'], 
    password=os.environ['mongodb_password']
)

def llm_retriever(llm, query, database_name):
    retriever_prompt = read_yaml('./prompts/collection_retriever_prompt.yml')
    system_message = retriever_prompt['system_message']
    prompt = retriever_prompt['prompt']

    df_desc = mongo.find_all(os.environ['mongodb_table_descriptor'], database_name)

    # if there is only 1 table in database, return 1 without going through the whole process
    if df_desc.shape[0] == 1:
        table_name = df_desc['collection'].iloc[0]
        description = df_desc['description'].iloc[0]
        return table_name, description
    elif df_desc.shape[0] == 0:
        raise RuntimeError('[Collection Retriever] No collection found')

    table_descs = []
    for idx, row in df_desc.iterrows():
        table_desc = f'```{row["collection"]}\n{row["description"]}\n```'
        table_descs.append(table_desc)

    table_descs = '\n\n'.join(table_descs)
    prompt = prompt.format(query=query, table_descriptions=table_descs)
    response = llm.llm_runnable.invoke({'system_message': system_message, 'prompt': prompt.format(query=query)})
    table_name = response.strip()
    # table_name = table_name.split('```')[1]

    description = df_desc.loc[df_desc['collection'] == table_name, 'description'].iloc[0]

    return table_name, description

if __name__ == '__main__':
    from utils.llm import LargeLanguageModel
    model_config = read_yaml('./model_configs/leoscorpius.yml')
    llm = LargeLanguageModel(**model_config)

    tb_name, desc = llm_retriever(
        llm,
        'How many unique debtors are there?',
        'troin_test'
    )

    print(tb_name)
    print(desc)