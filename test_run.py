from utils.llm import LargeLanguageModelAgent, LargeLanguageModel
from utils.mongoDB import MongoDBController
from langchain.globals import set_verbose
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from dotenv import load_dotenv

import os
import sys
import yaml
import time
import argparse
import traceback
import wandb
import datetime
import langchain
import logging

import numpy as np
import pandas as pd

load_dotenv('./.env.development')

# set_verbose(False)
langchain.debug = True

ap = argparse.ArgumentParser()
ap.add_argument('--save_path', type=str, required=True)
ap.add_argument('--evaluator', type=str, default='./model_configs/neural-chat.yml')
ap.add_argument('--model', type=str, default='./model_configs/neural-chat.yml')
ap.add_argument('--prompt', type=str, default='./prompts/pandas_prompt_01.yml')
ap.add_argument('--use-custom-prompt', action='store_true', default=False)
ap.add_argument('--questions_answers', type=str, default='./testing/questions_answers.xlsx')
ap.add_argument('--log-file-path', type=str, default='./logs/debug.log')
ap.add_argument('--wandb_project', type=str, default='langchain-tracing')
args = ap.parse_args()

# Redirect stderr to the logging module
sys.stderr = logging.StreamHandler(sys.stdout).stream

if args.log_file_path:
    os.makedirs('logs', exist_ok=True)
    sys.stdout = open(args.log_file_path, 'w')
    sys.stderr = open(args.log_file_path, 'a')

save_folder = args.save_path.split('/')
save_folder = '/'.join(save_folder[:-1])
os.makedirs(save_folder, exist_ok=True)

os.environ['LANGCHAIN_WANDB_TRACING'] = 'true'

mongo = MongoDBController(
    host=os.environ['mongodb_url'],
    port=int(os.environ['mongodb_port']), 
    username=os.environ['mongodb_user'], 
    password=os.environ['mongodb_password']
)

with open(args.prompt, 'r') as f:
    prompt_config = yaml.safe_load(f)

with open('./prompts/evaluate_prompt.yml', 'r') as f:
    evaluate_prompt_template = yaml.safe_load(f)

llm_agent = LargeLanguageModelAgent(args.model)
llm = llm_agent.llm
llm.load_prefix_suffix(
    prefix_text=prompt_config['prefix'],
    suffix_text=prompt_config['suffix']
)

if args.model != args.evaluator:
    with open(args.evaluator, 'r') as f:
        evaluator_config = yaml.safe_load(f)
        evaluator_llm = LargeLanguageModel(**evaluator_config)
else:
    evaluator_llm = llm

df_questions = pd.read_excel(args.questions_answers, sheet_name=None)

if args.use_custom_prompt:
    prefix = llm.prefix
    suffix = llm.suffix
    include_df_in_prompt = True
else:
    prefix = None
    suffix = None
    include_df_in_prompt = True

writer = pd.ExcelWriter(args.save_path)
for dataset_name, df_question in df_questions.items():
    df = mongo.find_all(db_name='test_data', collection_name=dataset_name)
    table_desc = mongo.get_table_desc('test_data', dataset_name)
    
    dataframe_agent = llm_agent.create_dataframe_agent(df, table_desc)
    results = []
    wandb_run_name = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M")} - {dataset_name}'
    with wandb.init(project=args.wandb_project, name=wandb_run_name) as run:
        for idx, row in df_question.iterrows():
            question = row['Questions']
            answer = str(row['Answers'])
            start = time.time()
            try:
                result = dataframe_agent({'input': question})
                end = time.time()

                llm_output = str(result['output'])

                intermediate_steps = result['intermediate_steps']

                # evaluation
                evaluation_prompt = evaluate_prompt_template['user'].format(
                    query=question,
                    ground_truth=answer,
                    answer=llm_output
                )
                evaluation_prompt = PromptTemplate.from_template(
                    evaluation_prompt
                )
                runnable = evaluation_prompt | evaluator_llm.llm | StrOutputParser()
                rating = runnable.invoke({'system_message': evaluate_prompt_template['system'], 'prompt': evaluation_prompt})
            except:
                end = time.time()
                intermediate_steps = None
                llm_output = traceback.format_exc()
                rating = -1

            results.append({
                'question': question,
                'execution_time': np.round(end-start, 2),
                'llm_output': llm_output,
                'answer': answer,
                'rating': int(rating),
                'debug': intermediate_steps,
                'final_prompt': dataframe_agent.agent.llm_chain.prompt
            })

    df_result = pd.DataFrame(results)
    df_result.to_excel(writer, index=False, sheet_name=dataset_name)

writer.close()

# The accuracy for each dataset and overall accuracy will be printed inside './log/logfile.txt' (bottom)
dfs = pd.read_excel(args.save_path, sheet_name=None)
overall_rating = []
df_rating = []
for dataset_name, df in dfs.items():
    rating = df['rating']
    score = rating.mean()
    df_rating.append({'dataset': dataset_name, 'accuracy': f'{score:.2f}'})
    overall_rating += rating.tolist()

df_rating = pd.DataFrame(df_rating)
overall_rating = np.mean(overall_rating)
print(f'Overall accuracy: {overall_rating:.2f}')
print(df_rating.to_markdown())