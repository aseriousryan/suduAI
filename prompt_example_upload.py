from utils.mongoDB import MongoDBController
from fastapi.responses import JSONResponse
from fastapi import HTTPException, FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from preprocessors.prompt_descriptor import store_question_embedding
import pandas as pd
import json
from bson import json_util
import os
import uvicorn
import traceback
from utils.common import ENV


load_dotenv(f'./.env.{ENV}')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mongo = MongoDBController(
    host=os.environ['mongodb_url'],
    port=int(os.environ['mongodb_port']), 
    username=os.environ['mongodb_user'], 
    password=os.environ['mongodb_password']
)

@app.get('/')
async def root():
    with open('./version.md', 'r') as f:
        version = f.read()
    content = {
        'version': version,
        'env': ENV
    }

    return JSONResponse(content=content)


@app.post('/upload')
def upload(
    file: UploadFile,
    collection_name: str
):
    try:
        contents = file.file.read()
        with open (file.filename, 'wb') as f:
            f.write(contents)

        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.filename)

        # add new columns to the DataFrame
        df['question_token_length'] = None
        df['question_embedding'] = None

        # add question embeddings to database
        for index, row in df.iterrows():
            _, ques_token_length, emb = store_question_embedding(row['question'])
            df.at[index, 'question_token_length'] = ques_token_length
            df.at[index, 'question_embedding'] = emb 

        data_dict = df.to_dict("records")
        inserted_ids = mongo.insert_many(data_dict, os.environ['mongodb_prompt_example_descriptor'], collection_name)
        inserted_ids = json.loads(json_util.dumps(inserted_ids))
        os.remove(file.filename)

        return JSONResponse(content={"insert_id": inserted_ids})

    except:
        raise HTTPException(status_code=404, detail=traceback.format_exc())
    

if __name__ == "__main__":
    uvicorn.run('prompt_example_upload:app', host="0.0.0.0", port=8082, reload=True)
