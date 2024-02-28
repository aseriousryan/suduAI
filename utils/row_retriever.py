import sys
sys.path.append('.')

from utils.common import ENV
from utils.mongoDB import MongoDBController
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import os

load_dotenv(f'./.env.{ENV}')

mongo = MongoDBController(
    host=os.environ['mongodb_url'],
    port=int(os.environ['mongodb_port']), 
    username=os.environ['mongodb_user'], 
    password=os.environ['mongodb_password']
)
print(os.environ['mongodb_url'])

#model = BGEM3FlagModel(os.environ['row_embedding_model'], use_fp16=True)

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

db_name = 'de_carton'
collection_name = 'delivery_delivery_order_listing'

data_frame = mongo.find_all(db_name=db_name, collection_name=collection_name, exclusion={'_id': 0})

def top5_row_for_question(question, data_frame):
    # Encode the user question
    question_emb = model.encode([question], return_dense=True)['dense_vecs'][0]

    similarity_scores = []

    for idx, doc in data_frame.iterrows():
        doc_embedding = doc['row_embedding']  # Adjust based on your actual field name
        score = cosine_similarity([question_emb], [doc_embedding])[0][0]
        similarity_scores.append(score)  # Store document ID and score

    # Add a new column 'similarity_score' to the data_frame
    data_frame['similarity_score'] = similarity_scores

    # Sort dataframe by similarity score in descending order and reset index
    data_frame_sorted = data_frame.sort_values(by='similarity_score', ascending=False).reset_index(drop=True)

    top_rows = data_frame_sorted.head(5).drop(columns=['row_embedding'], errors='ignore')

    return top_rows

if __name__ == '__main__':
    print(str(top5_row_for_question("What is the total delivery order for Totonku?", data_frame).to_markdown()))