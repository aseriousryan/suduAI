from utils.llm import LargeLanguageModel
from utils.mongoDB import MongoDBController
from utils.redirect_print import RedirectPrint
from utils.common import (
    tokenize, ENV, read_yaml, parse_langchain_debug_log, LogData
)

from langchain.globals import set_debug
from fastapi.responses import JSONResponse
from fastapi import HTTPException, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from aserious_agent.pandas_agent import PandasAgent

from aserious_agent.sql_agent import SQLAgent
from tools.sql_database_toolkit import DatabaseConnection
from langchain.sql_database import SQLDatabase

import os
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

# initialize modules
rp = RedirectPrint()
data_logger = LogData()
llm = LargeLanguageModel(**read_yaml(os.environ['model']))
mongo = MongoDBController(
    host=os.environ['mongodb_url'],
    port=int(os.environ['mongodb_port']), 
    username=os.environ['mongodb_user'], 
    password=os.environ['mongodb_password']
)


db_user = os.getenv('db_user')
db_password = os.getenv('db_password')
db_host = os.getenv('db_host')
db_name = os.getenv('db_name')
db_port = int(os.getenv('db_port'))

db_connection = DatabaseConnection(db_user, db_password, db_host, db_name, db_port)

engine = db_connection.get_engine()

sql_db = SQLDatabase(engine)

agent_type = os.getenv('agent_type')

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

        if(agent_type=="sql_agent"):
            agent = SQLAgent(llm, data_logger, mongo, sql_db)
        else:
            agent = PandasAgent(llm, mongo, data_logger)

        # capture terminal outputs to log llm output
        rp.start()
        result = agent.run_agent(user_query=msg, database_name=database_name, collection=collection)
        debug_log = parse_langchain_debug_log(rp.get_output())
        rp.stop()
        
        n_token_output = len(tokenize(os.environ['tokenizer'], debug_log))

        error_message = "Agent stopped due to iteration limit or time limit"

        # log data to mongodb
        success = error_message not in result.get('output')
        data = data_logger.model_dump()
        data.update({
            'datetime': now,
            'query': msg,
            'output': result.get('output'),
            'logs': debug_log,
            'n_token_output': n_token_output,
            'database_name': database_name,
            'note': note,
            'success': success
        })

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
        data = data_logger.model_dump()
        data.update({
            'datetime': datetime.datetime.now(pytz.timezone('Asia/Singapore')),
            'query': msg,
            'error': traceback.format_exc(),
            'note': note,
            'success': False
        })

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