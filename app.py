from langchain.globals import set_verbose
import pandas as pd
import datetime
import pytz
import fastapi
import uvicorn
from LLM import LargeLanguageModelAgent
from MongoDB import MongoDBOperations
from langchain.globals import set_verbose
import fastapi
import uvicorn
import pandas as pd

set_verbose(True)
app = fastapi.FastAPI()

llm_agent = LargeLanguageModelAgent('./model_configs/neural-chat.yml', './prompts/pandas_prompt_04.yml')
mongo_ops = MongoDBOperations('quincy.lim-everest.nord', 27017, '', '27017')

@app.post('/store')
async def store(companyName, csv_file):
    df = pd.read_csv(f"./data/csv_data/{csv_file}.csv")
    data_dict = df.to_dict("records")
    mongo_ops.insert_many(companyName, csv_file, data_dict)

@app.post('/chat')
async def chatmsg(msg, database_name, collection):
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
    return result.get('output')

if __name__ == "__main__":
    uvicorn.run('app:app', host="0.0.0.0", port=8082, reload=False)
