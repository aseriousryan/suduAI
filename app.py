from langchain.globals import set_verbose
import pandas as pd
import datetime
import pytz
import fastapi
import uvicorn
from utils.llm import LargeLanguageModelAgent
from utils.mongoDB import MongoDBOperations
from langchain.globals import set_verbose
import fastapi
import uvicorn
import pandas as pd
import yaml

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
        print(f"Error in /store: {e}")

@app.post('/chat')
async def chatmsg(msg, database_name, collection):
    try:
        data = mongo_ops.find_all(database_name, collection)
        dataframe_agent = llm_agent.create_dataframe_agent(data)
        result = dataframe_agent({'input': msg})

        #evaluation table (hard-coded)
        data =  {
            'datetime': datetime.datetime.now(pytz.timezone('Asia/Singapore')),
            'query': msg,
            'output': result.get('output')
        }
        mongo_ops.insert_one(data)

         # Return a dictionary with the result
        return {'result': result}
    except Exception as e:
        print(f"Error in /chat: {e}")

if __name__ == "__main__":
    try:
        uvicorn.run('app:app', host="0.0.0.0", port=8082, reload=False)
    except Exception as e:
        print(f"Error running the app: {e}")

