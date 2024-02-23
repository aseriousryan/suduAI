from llm import LargeLanguageModel
from aserious_agent.pandas_agent import create_pandas_dataframe_agent
from langchain.globals import set_verbose
from pymongo import MongoClient

import pandas as pd

import datetime
import pytz
import yaml
import fastapi
import uvicorn

set_verbose(True)
app = fastapi.FastAPI()

client = MongoClient('quincy.lim-everest.nord', port=27017, username='', password='27017')
db = client['logs']
collection = db['de-carton']

with open('./model_configs/neural-chat.yml', 'r') as f:
    model_config = yaml.safe_load(f)

with open('./prompts/pandas_prompt_04.yml', 'r') as f:
    prompt_config = yaml.safe_load(f)

llm = LargeLanguageModel(**model_config, **prompt_config)

df = pd.read_csv('./data/csv_data/clean_data.csv')
dataframe_agent = create_pandas_dataframe_agent(
    llm.llm, 
    df,
    verbose=True,
    prefix=llm.prefix,
    suffix=llm.suffix,
    input_variables=['input', 'agent_scratchpad', 'df_head'],
    agent_executor_kwargs={'handle_parsing_errors': True},
    include_df_in_prompt=True,
    return_intermediate_steps=True,
    max_iterations=10,
    max_execution_time=600,
    early_stopping_method='force', 
)

@app.post('/chat')
async def chat(msg):
    result = dataframe_agent({'input': msg})
    collection.insert_one({
        'datetime': datetime.datetime.now(pytz.timezone('Asia/Singapore')),
        'query': msg,
        'output': result.get('output')
    })

    return result.get('output')


if __name__ == "__main__":
    uvicorn.run('demo:app', host="0.0.0.0", port=8081, reload=False)
