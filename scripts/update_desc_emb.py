import sys
sys.path.append('.')

import os
import argparse
import datetime

from sentence_transformers import SentenceTransformer
from utils.mongoDB import MongoDBController
from dotenv import load_dotenv

ap = argparse.ArgumentParser()
ap.add_argument('--env', type=str, help='<production|development>', default='development')
ap.add_argument('--model', type=str, help='path to sentence transformer model')
args = ap.parse_args()

load_dotenv(f'./.env.{args.env}')

mongo = MongoDBController(
    host=os.environ['mongodb_url'],
    port=int(os.environ['mongodb_port']), 
    username=os.environ['mongodb_user'], 
    password=os.environ['mongodb_password']
)

model = SentenceTransformer(args.model, device='cuda')
collections = mongo.list_collections(os.environ['mongodb_table_descriptor'])

for collection in collections:
    print(f'[*] Updating {collection}')
    df = mongo.find_all(os.environ['mongodb_table_descriptor'], collection)
    for idx, row in df.iterrows():
        if 'retrieval_description' not in row.index or isinstance(row['retrieval_description'], float):
            desc = row['description']
        else:
            desc = row['retrieval_description']
        desc = desc.split('\nThis is the quantitative information ')[0].strip()
        emb = model.encode(desc, convert_to_numpy=True).tolist()
        query_criteria = {'_id': row['_id']}
        update_data = {'$set': {'embedding': emb, 'datetime': datetime.datetime.now()}}
        mongo.collection.update_one(query_criteria, update_data)