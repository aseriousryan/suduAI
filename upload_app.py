from utils.mongoDB import MongoDBController
from fastapi.responses import JSONResponse
from fastapi import HTTPException, FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from preprocessors import cv_de_carton
from dotenv import load_dotenv

import pandas as pd
import os
import uvicorn
import yaml
import traceback

load_dotenv('./.env')

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

    return JSONResponse(content=version)


@app.post('/upload')
def upload(file: UploadFile, uuid, collection_name, preprocess=False):
    try:
        contents = file.file.read()
        with open (file.filename, 'wb') as f:
            f.write(contents)

        if preprocess:
            image = cv_de_carton.convert_pdf_to_image(file.filename, 500)
            cv_de_carton.create_borders(image, uuid)
            df = cv_de_carton.extract_table(f"{uuid}.jpg", uuid)
            os.remove(f"{uuid}.jpg")
            os.remove(f"{uuid}.csv")
        else:
           df = pd.read_csv(file.filename)

        data_dict = df.to_dict("records")
        response = mongo.insert_many(data_dict, uuid, collection_name)
        os.remove(file.filename)
        

        print(response)
  

    except:
        raise HTTPException(status_code=404, detail=traceback.format_exc())

    

if __name__ == "__main__":
    uvicorn.run('upload_app:app', host="0.0.0.0", port=8082, reload=False)
