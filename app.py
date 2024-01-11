from utils.llm import LargeLanguageModelAgent
from utils.mongoDB import MongoDBController
from utils.redirect_print import RedirectPrint
from utils.common import tokenize, ENV, read_yaml
from utils.collection_retriever import llm_retriever
from langchain.globals import set_verbose
from fastapi.responses import JSONResponse
from fastapi import HTTPException, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import pandas as pd

import time
import os
import uvicorn
import yaml
import pytz
import datetime
import traceback

load_dotenv(f'./.env.{ENV}')

set_verbose(True)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
rp = RedirectPrint() 

llm_agent = LargeLanguageModelAgent(os.environ['model'])
mongo = MongoDBController(
    host=os.environ['mongodb_url'],
    port=int(os.environ['mongodb_port']), 
    username=os.environ['mongodb_user'], 
    password=os.environ['mongodb_password']
)

# Load YAML files using paths from environment variables
prompt = os.environ.get('prompt')
prompt_template = read_yaml(prompt)

# Accessing prefix and suffix
prefix = prompt_template['prefix']
suffix = prompt_template['suffix']

@app.get('/')
async def root():
    with open(os.environ['model'], 'r') as f: 
        model_config = yaml.safe_load(f)

    with open('./version.md', 'r') as f:
        version = f.read()

    model_config['API_version'] = version 

    return JSONResponse(content=model_config)

@app.post('/chat')
async def chatmsg(msg: str, database_name: str, collection: str = None):
    # database_name = company name
    try:
        llm_agent.llm.load_prefix_suffix(prefix, suffix)

        # retrieve collection
        start = time.time()
        if collection is None:
            collection, table_desc = llm_retriever(llm_agent.llm, msg, database_name)
        end = time.time()
        time_collection_retrieval = end - start

        data = mongo.find_all(database_name, collection, projection={'_id': 0})
        if data.shape[0] == 0:
            raise RuntimeError(f'No data found:\ndb: {database_name}\ncollection: {collection}')

        dataframe_agent = llm_agent.create_dataframe_agent(data, table_desc)

        # capture terminal outputs to log llm output
        rp.start()
        start = time.time()
        result = dataframe_agent({'input': msg})
        output_log = rp.get_output().split('Prompt after formatting:')[-1]
        end = time.time()
        rp.stop()

        n_token_output = len(tokenize(os.environ['tokenizer'], output_log))

        success = True
        # Check for specific error message in the output
        if "Agent stopped due to iteration limit or time limit" in result.get('output'):
            success = False
            data =  {
                'datetime': datetime.datetime.now(pytz.timezone('Asia/Singapore')),
                'query': msg,
                'output': result.get('output'),
                'logs': output_log,
                'n_token_output': n_token_output,
                'response_time': end - start,
                'collection_retrieval_time': time_collection_retrieval,
                'collection': collection,
                'database_name': database_name,
                'success': success
            }
            id = mongo.insert_one(
                data=data,
                db_name=os.environ['mongodb_log_db'],
                collection_name=database_name
            )

            error_detail = str({'error':'Agent stopped due to iteration limit or time limit.', 'mongo_id': str(id)})
            raise HTTPException(status_code=405, detail=error_detail)
        
        data =  {
            'datetime': datetime.datetime.now(pytz.timezone('Asia/Singapore')),
            'query': msg,
            'output': result.get('output'),
            'logs': output_log,
            'n_token_output': n_token_output,
            'response_time': end - start,
            'collection_retrieval_time': time_collection_retrieval,
            'collection': collection,
            'database_name': database_name,
            'success': success
        }

        # log 
        id = mongo.insert_one(
            data=data,
            db_name=os.environ['mongodb_log_db'],
            collection_name=database_name
        )

         # Return a dictionary with the result
        return {'result': result.get('output'), 'mongo_id': str(id)}
    except HTTPException as e:
        # If the exception is an HTTPException, just raise it
        raise e
    except:
        data =  {
            'datetime': datetime.datetime.now(pytz.timezone('Asia/Singapore')),
            'query': msg,
            'error': traceback.format_exc(),
            'success': False
        }

        if 'output_log' in globals():
            data['logs'] = output_log

        id = mongo.insert_one(
            data=data,
            db_name=os.environ['mongodb_log_db'],
            collection_name=database_name
        )
        
        error_detail = str({'error':traceback.format_exc(), 'mongo_id': str(id)})
        raise HTTPException(status_code=404, detail=error_detail)

if __name__ == "__main__":
    uvicorn.run('app:app', host="0.0.0.0", port=8080, reload=False)
