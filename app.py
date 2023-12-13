from utils.llm import LargeLanguageModelAgent
from utils.mongoDB import MongoDBOperations
from utils.redirect_print import RedirectPrint
from langchain.globals import set_verbose
from fastapi.responses import JSONResponse
from fastapi import HTTPException, FastAPI
from dotenv import load_dotenv

import pandas as pd
import sentencepiece as spm

import os
import uvicorn
import yaml
import pytz
import datetime
import traceback

load_dotenv('./.env')

set_verbose(True)
app = FastAPI()
rp = RedirectPrint()
sp = spm.SentencePieceProcessor(model_file=os.environ['tokenizer'])

try:
    llm_agent = LargeLanguageModelAgent(os.environ['model'], os.environ['prompt_template'])
    mongo_ops = MongoDBOperations(
        host=os.environ['mongodb_url'],
        port=int(os.environ['mongodb_port']), 
        username=os.environ['mongodb_user'], 
        password=os.environ['mongodb_password']
    )
except:
    raise RuntimeError(f'Error initializing: {traceback.format_exc()}')

@app.get('/')
async def root():
    with open(os.environ['model'], 'r') as f:
        model_config = yaml.safe_load(f)
    return JSONResponse(content=model_config)

@app.post('/store')
async def store(company_name, csv_file):
    try:
        df = pd.read_csv(f"./data/csv_data/{csv_file}.csv")
        data_dict = df.to_dict("records")
        mongo_ops.insert_many(company_name, csv_file, data_dict)
    except:
        raise HTTPException(status_code=404, detail=traceback.format_exc())

@app.post('/chat')
async def chatmsg(msg, database_name, collection):
    try:
        data = mongo_ops.find_all(database_name, collection)
        dataframe_agent = llm_agent.create_dataframe_agent(data)

        rp.start()
        result = dataframe_agent({'input': msg})
        output_logs = rp.get_output()
        rp.stop()

        n_token_output = len(sp.encode(output_logs))

        #evaluation table (hard-coded)
        data =  {
            'datetime': datetime.datetime.now(pytz.timezone('Asia/Singapore')),
            'query': msg,
            'output': result.get('output'),
            'logs': output_logs,
            'n_token_output': n_token_output

        }
        mongo_ops.insert_one(data)

         # Return a dictionary with the result
        return {'result': result.get('output')}
    except:
        raise HTTPException(status_code=404, detail=traceback.format_exc())

if __name__ == "__main__":
    uvicorn.run('app:app', host="0.0.0.0", port=8082, reload=False)
