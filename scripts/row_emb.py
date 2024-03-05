import sys
sys.path.append('/home/seriousco/Documents/jiaxin/suduAI-1')

import os
from FlagEmbedding import BGEM3FlagModel
from utils.mongoDB import MongoDBController
from dotenv import load_dotenv


load_dotenv('/home/seriousco/Documents/jiaxin/suduAI-1/.env.production')

mongo = MongoDBController(
    host=os.environ['mongodb_url'],
    port=int(os.environ['mongodb_port']), 
    username=os.environ['mongodb_user'], 
    password=os.environ['mongodb_password']
)
print(os.environ['mongodb_url'])

model = BGEM3FlagModel("/home/seriousco/Documents/jiaxin/suduAI-1/models/bge-m3", use_fp16=True)

def compute_embedding(row):
    # Exclude '_id' from the row
    without_id = {key: str(value) for key, value in row.items() if key != '_id'}
    
    print(without_id) 

    # Convert the dictionary values to a list of strings and join them
    row_string = ' '.join(without_id.values())
    
    print(row_string)

    # Row Embedding 
    row_emb = model.encode([row_string], return_dense=True)['dense_vecs']

    # Embedded Row
    return row_emb.tolist()

db = "de_carton" # Change the DB name

collection= "supplier_aging_report" # Change the collection name

rows = mongo.find_all(db, collection)

for idx, row in rows.iterrows():

    row_dict = row.to_dict() # convert to dictionary

    row_id = row_dict.pop('_id', None)  # Extract and remove the '_id' field
    
    # Compute the embedding for the row, without '_id' field
    emb = compute_embedding(row_dict)

    # Avoid nested list
    emb_flat = emb[0] 

    # current row '_id'
    query_criteria = {'_id': row_id}
    
    # new data: row embedding
    update_data = {'$set': {'row_embedding': emb_flat}}
    
    # Update the document in the MongoDB collection
    mongo.collection.update_one(query_criteria, update_data)

    print("added")