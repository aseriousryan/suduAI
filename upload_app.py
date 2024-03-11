from utils.mongoDB import MongoDBController
from fastapi.responses import JSONResponse
from fastapi import HTTPException, FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from preprocessors import cv_de_carton, troin, table_descriptor, row_descriptor, table_schema
from dotenv import load_dotenv
import json
from bson import json_util
from utils.common import convert_to_date, ENV, convert_date_columns_to_sql_datetime


import pandas as pd
import os
import uvicorn
import yaml
import traceback
import datetime

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
print(os.environ['mongodb_url'])

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
    uuid: str,
    collection_name: str,
    preprocess: bool = False,
    desc: str = '',
    retrieval_desc: str = None,
):
    try:
        contents = file.file.read()
        with open(file.filename, 'wb') as f:
            f.write(contents)

        if file.filename.endswith('.xlsx'):
            if preprocess:
                df = troin.troin_sales(file.filename)
            else:
                df = pd.read_excel(file.filename)
    
        elif file.filename.endswith('.pdf'):
            image = cv_de_carton.convert_pdf_to_image(file.filename, 500)
            cv_de_carton.create_borders(image, uuid)
            df = cv_de_carton.extract_table(f"{uuid}.jpg", uuid)
            os.remove(f"{uuid}.jpg")
            os.remove(f"{uuid}.csv")

        elif file.filename.endswith('.csv'):
            df = pd.read_csv(file.filename)
   
        df = convert_date_columns_to_sql_datetime(df)

        inserted_ids = mongo.insert_unique_rows(df, uuid, collection_name)

        inserted_ids = json.loads(json_util.dumps(inserted_ids))

        # Remove the uploaded file
        os.remove(file.filename)

        description_df, description_length, description_emb = table_descriptor.get_table_description(df, desc, retrieval_desc)
        mongo.create_database(os.environ['mongodb_table_descriptor'])
        
        mongo.create_collection(uuid)

        # Check if a document with the same collection name already exists in the database
        existing_collection = mongo.get_table_desc_collection(uuid,collection_name)
  
        if existing_collection.empty:
            table_desc_id = mongo.insert_one({
                'collection': collection_name, 
                'description': description_df,
                'token_length': description_length,
                'embedding': description_emb,
                'retrieval_description': retrieval_desc if retrieval_desc is not None else description_df,
                'datetime': datetime.datetime.now()
            })
            table_desc_id = str(table_desc_id)
        else:
            table_desc_id = "Collection name already exists in the database."
     

        # add table schema to database
        columns_str, data_types_str, retrieval_desc, desc_token_length, emb = table_schema.get_table_schema(df, retrieval_desc)
        mongo.create_database(os.environ['mongodb_sql_table_schema'])
        
        # the collection name in description database is the uuid
        mongo.create_collection(uuid)

        # Check if a document with the same collection name already exists in the database
        existing_table = mongo.get_table_schema_collection(uuid, collection_name)

        if existing_table.empty:
            # If the DataFrame is empty, there is no existing table with the given name
            table_schema_id = mongo.insert_one({
                'table_name': collection_name, 
                'column_name': columns_str,
                'data_type': data_types_str,
                'retrieval_description': retrieval_desc if retrieval_desc is not None else "No desc",
                'embedding': emb,
                'token_length': desc_token_length,
                'datetime': datetime.datetime.now()
            })
            table_schema_id = str(table_schema_id)
        else:
            table_schema_id = "Table name already exists in the database."

        return JSONResponse(content={"insert_id": inserted_ids,"table_desc_id": table_desc_id,  "table_schema_id":table_schema_id})

    except:
        raise HTTPException(status_code=404, detail=traceback.format_exc())


if __name__ == "__main__":
    uvicorn.run('upload_app:app', host="0.0.0.0", port=8082, reload=True)



    