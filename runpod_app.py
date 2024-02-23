from utils.llm import LargeLanguageModelAgent
from utils.mongoDB import MongoDBController
from utils.redirect_print import RedirectPrint
from utils.common import tokenize
from langchain.globals import set_verbose
from dotenv import load_dotenv

import pandas as pd

import time
import os
import uvicorn
import yaml
import pytz
import datetime
import traceback
import runpod

load_dotenv('./.env')

set_verbose(True)
rp = RedirectPrint()

llm_agent = LargeLanguageModelAgent(os.environ['model'])
mongo = MongoDBController(
    host=os.environ['mongodb_url'],
    port=int(os.environ['mongodb_port']), 
    username=os.environ['mongodb_user'], 
    password=os.environ['mongodb_password']
)

def load_prompt_prefix_suffix(company_name):
    db_name = os.environ['mongodb_prompt_db']
    if not mongo.is_collection_exist(company_name, db_name):
        # load default prompt if company name not in prompt database
        df_prompt = mongo.find_all(db_name, 'default')
    else:
        df_prompt = mongo.find_all(db_name, company_name)

    latest_prompt = df_prompt[df_prompt['date'] == df_prompt['date'].max()].iloc[0]

    return latest_prompt['prefix'], latest_prompt['suffix']

def root(job):
    with open(os.environ['model'], 'r') as f:
        model_config = yaml.safe_load(f)

    with open('./version.md', 'r') as f:
        version = f.read()

    model_config['API_version'] = version

    return model_config


def store(company_name, csv_file):
    try:
        df = pd.read_csv(f"./data/csv_data/{csv_file}.csv")
        data_dict = df.to_dict("records")
        mongo.insert_many(company_name, csv_file, data_dict)
    except:
        raise {'error': traceback.format_exc()}

def chatmsg(job):
    job = job['input']
    database_name = job['database_name']
    collection = job['collection']
    msg = job['msg']
    # database_name = company name
    try:
        prefix, suffix = load_prompt_prefix_suffix(database_name)
        llm_agent.llm.load_prefix_suffix(prefix, suffix)
        data = mongo.find_all(database_name, collection)
        if data.shape[0] == 0:
            raise RuntimeError(f'No data found:\ndb: {database_name}\ncollection: {collection}')

        dataframe_agent = llm_agent.create_dataframe_agent(data)

        # capture terminal outputs to log llm output
        rp.start()
        start = time.time()
        result = dataframe_agent({'input': msg})
        output_log = rp.get_output()
        end = time.time()
        rp.stop()

        n_token_output = len(tokenize(os.environ['tokenizer'], output_log))

        data =  {
            'datetime': datetime.datetime.now(pytz.timezone('Asia/Singapore')),
            'query': msg,
            'output': result.get('output'),
            'logs': output_log,
            'n_token_output': n_token_output,
            'time': end - start,
            'success': True
        }

        # log 
        mongo.insert_one(
            data=data,
            db_name=os.environ['mongodb_log_db'],
            collection_name=database_name
        )

         # Return a dictionary with the result
        return {'result': result.get('output')}
    except:
        data =  {
            'datetime': datetime.datetime.now(pytz.timezone('Asia/Singapore')),
            'query': msg,
            'error': traceback.format_exc(),
            'success': False
        }

        if 'output_log' in globals():
            data['logs'] = output_log

        mongo.insert_one(
            data=data,
            db_name=os.environ['mongodb_log_db'],
            collection_name=database_name
        )

        raise RuntimeError(traceback.format_exc())
    
runpod.serverless.start({'handler': chatmsg})
