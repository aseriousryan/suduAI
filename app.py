from utils.llm import LargeLanguageModel
from utils.mongoDB import MongoDBController
from utils.redirect_print import RedirectPrint
from utils.common import tokenize, ENV, read_yaml, parse_langchain_debug_log
from utils.collection_retriever import sentence_transformer_retriever
from utils.prompt_retriever import prompt_example_sentence_transformer_retriever
from utils.row_retriever import top5_row_for_question
from langchain.globals import set_debug
from fastapi.responses import JSONResponse
from fastapi import HTTPException, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from aserious_agent.pandas_agent import PandasAgent

import pandas as pd

import time
import os
import re
import uvicorn
import yaml
import pytz
import datetime
import traceback

load_dotenv(f'./.env.{ENV}')

set_debug(True)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
rp = RedirectPrint() 

llm = LargeLanguageModel(**read_yaml(os.environ['model']))
mongo = MongoDBController(
    host=os.environ['mongodb_url'],
    port=int(os.environ['mongodb_port']), 
    username=os.environ['mongodb_user'], 
    password=os.environ['mongodb_password']
)

@app.get('/')
async def root():
    with open(os.environ['model'], 'r') as f: 
        model_config = yaml.safe_load(f)

    with open('./version.md', 'r') as f:
        version = f.read()

    model_config['API_version'] = version 
    model_config['env'] = ENV

    return JSONResponse(content=model_config)

@app.post('/chat')
async def chatmsg(msg: str, database_name: str, collection: str = None, note: str = None):
    # database_name = company name
    try:
        now = datetime.datetime.now()

        # retrieve collection
        start = time.time()
        if collection is None:
            collection, table_desc, desc_cos_sim = sentence_transformer_retriever(msg, database_name)
        else:
            table_desc = mongo.get_table_desc(database_name, collection)
            desc_cos_sim = -1

        prompt_example, question_retrieval = prompt_example_sentence_transformer_retriever(msg, database_name)
        end = time.time()
        time_collection_retrieval = end - start

        data = mongo.find_all(database_name, collection, exclusion={'_id': 0})
        if data.shape[0] == 0:
            raise RuntimeError(f'No data found:\ndb: {database_name}\ncollection: {collection}')

        # Row Embedding
        top5_results = str(top5_row_for_question(msg, data).to_markdown())
        
        agent = PandasAgent(llm, data).create_agent(
            prompt_example=prompt_example,
            table_desc=table_desc,
            df_head=top5_results
        )

        # capture terminal outputs to log llm output
        rp.start()
        start = time.time()
        result = agent.invoke({'input': msg})
        debug_log = parse_langchain_debug_log(rp.get_output())
        end = time.time()
        rp.stop()
        
        n_token_output = len(tokenize(os.environ['tokenizer'], debug_log))

        error_message = "Agent stopped due to iteration limit or time limit"

        success = error_message not in result.get('output')
        data =  {
            'datetime': now,
            'query': msg,
            'output': result.get('output'),
            'logs': debug_log,
            'n_token_output': n_token_output,
            'response_time': end - start,
            'collection_retrieval_time': time_collection_retrieval,
            'collection': collection,
            'database_name': database_name,
            'desc_cos_sim': desc_cos_sim,
            'note': note,
            'query_retrieval': question_retrieval,
            'success': success
        }

        id = mongo.insert_one(
            data=data,
            db_name=os.environ['mongodb_log_db'],
            collection_name=database_name
        )

        if not success:
            error_detail = {'error': error_message, 'mongo_id': str(id)}
            return HTTPException(status_code=405, detail=error_detail)

        return {'result': result.get('output'), 'mongo_id': str(id)}
    
    except Exception:
        data =  {
            'datetime': datetime.datetime.now(pytz.timezone('Asia/Singapore')),
            'query': msg,
            'error': traceback.format_exc(),
            'note': note,
            'success': False
        }

        if 'output_log' in globals():
            data['logs'] = debug_log

        id = mongo.insert_one(
            data=data,
            db_name = os.environ['mongodb_log_db'],
            collection_name = database_name
        )
        
        error_detail = {'error': traceback.format_exc(), 'mongo_id': str(id)}
        raise HTTPException(status_code=404, detail=error_detail)

if __name__ == "__main__":
    uvicorn.run('app:app', host="0.0.0.0", port=8085, reload=False)