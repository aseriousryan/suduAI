import sys
sys.path.append('../notebooks/')

import os
import datetime
import requests
import time

import pandas as pd
from tqdm import tqdm
from utils.mongoDB import MongoDBController

from dotenv import load_dotenv

load_dotenv('./.env.development')

mongo = MongoDBController(
    host=os.environ['mongodb_url'],
    port=int(os.environ['mongodb_port']), 
    username=os.environ['mongodb_user'], 
    password=os.environ['mongodb_password']
)

# Reading the questions and ground truth values from the Excel file
excel_file_path = 'prev_question.xlsx'
df = pd.read_excel(excel_file_path, engine='openpyxl')

# Assuming the columns are named 'Questions' and 'Ground Truth'
questions = df['Questions'].tolist()
ground_truths = df['ground truth'].tolist()  # Reading the ground truth values

# Running for LLM - Open Chat
url = 'http://192.168.1.105:8084/chat'
note = 'dataset_mistral_9'

# Initialize an empty DataFrame to append results
df_llm = pd.DataFrame()

for question in tqdm(questions, desc="Processing questions"):  
    params = {
        'msg': question,
        'database_name': 'de_carton',
        'note': f'{note}',
    }

    headers = {'accept': 'application/json'}

    try:
        response = requests.post(url, params=params, headers=headers)  
        response.raise_for_status()  

    except requests.RequestException as e:
        print(f"Request failed: {e}")
        continue  

# After all queries for a topic have been processed, retrieve and filter the DataFrame
df_topic = mongo.find_all(db_name='development_logs', collection_name='de_carton')
mask = df_topic['note'].str.startswith(f'{note}', na=False)
df_llm_topic = df_topic.loc[mask, ['query', 'output']]
df_llm = pd.concat([df_llm, df_llm_topic])

print(len(df_llm['output']))
print(len(questions))

# Ensure that the DataFrame has been correctly populated
if len(questions) != len(df_llm['output']):
    raise ValueError("The number of questions does not match the number of outputs")

# Create the final DataFrame
df_result = pd.DataFrame({
    'question': questions,
    'ground_truth': ground_truths,  # Add the ground truth values to the DataFrame
    'llm': df_llm['output'].tolist(),
})

# Save the final DataFrame to an Excel file
df_result.to_excel('dataset_mistral_9', index=False)
