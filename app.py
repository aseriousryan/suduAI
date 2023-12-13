from langchain.globals import set_verbose
import pandas as pd
import datetime
import pytz
import fastapi
import uvicorn
from utils.llm import LargeLanguageModelAgent
from utils.mongoDB import MongoDBOperations
import yaml
import traceback
from fastapi import HTTPException

set_verbose(True)
app = fastapi.FastAPI()

try:
    llm_agent = LargeLanguageModelAgent('./model_configs/neural-chat.yml', './prompts/pandas_prompt_04.yml')
    mongo_ops = MongoDBOperations('quincy.lim-everest.nord', 27017, '', '27017')
except Exception as e:
    print(f"Error initializing LargeLanguageModelAgent or MongoDBOperations: {e}")

@app.get('/')
async def root():
    with open('./model_configs/neural-chat.yml', 'r') as f:
        model_config = yaml.safe_load(f)
    return {"model_config": model_config}

@app.post('/store')
async def store(company_name, csv_file):
    try:
        df = pd.read_csv(f"./data/csv_data/{csv_file}.csv")
        data_dict = df.to_dict("records")
        mongo_ops.insert_many(company_name, csv_file, data_dict)
    except Exception as e:
        tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
        tb_str = "".join(tb_str)
        raise HTTPException(status_code=500, detail=tb_str)

@app.post('/chat')
async def chatmsg(query, database_name):
    try:
        data = mongo_ops.find_all(database_name)
        dataframe_agent = llm_agent.create_dataframe_agent(data)
        result = dataframe_agent({'input': query})

        #evaluation table (hard-coded)
        data =  {
            'datetime': datetime.datetime.now(pytz.timezone('Asia/Singapore')),
            'query': query,
            'output': result.get('output')
        }
        mongo_ops.insert_one(data)

         # Return a dictionary with the result
        return {'result': result}
    except Exception as e:
        tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
        tb_str = "".join(tb_str)
        raise HTTPException(status_code=500, detail=tb_str)

if __name__ == "__main__":
    uvicorn.run('app:app', host="0.0.0.0", port=8082, reload=True)


