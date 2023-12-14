from utils.llm import LargeLanguageModel
from utils.mongoDB import MongoDBController
from aserious_agent.pandas_agent import create_pandas_dataframe_agent
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

import numpy as np
import pandas as pd

load_dotenv()

set_verbose(False)

ap = argparse.ArgumentParser()
ap.add_argument('--save_path', type=str, required=True)
ap.add_argument('--evaluator', type=str, default='./model_configs/neural-chat.yml')
ap.add_argument('--model', type=str, default='./model_configs/neural-chat.yml')
ap.add_argument('--prompt', type=str, default='./prompts/pandas_prompt_01.yml')
ap.add_argument('--use-custom-prompt', action='store_true', default=False)
ap.add_argument('--questions_answers', type=str, default='../questions_answers.xlsx')
ap.add_argument('--log-file-path', type=str, default='./logs/logfile.txt')
ap.add_argument('--wandb_project', type=str, default='langchain-tracing')
args = ap.parse_args()

# os.makedirs('logs', exist_ok=True)
# sys.stdout = open(args.log_file_path, 'w')
# sys.stderr = open(args.log_file_path, 'a')

save_folder = args.save_path.split('/')
save_folder = '/'.join(save_folder[:-1])
os.makedirs(save_folder, exist_ok=True)

os.environ['LANGCHAIN_WANDB_TRACING'] = 'true'
# os.environ['WANB_PROJECT'] = 'tmp-llm'

with open(args.model, 'r') as f:
    model_config = yaml.safe_load(f)

with open(args.prompt, 'r') as f:
    prompt_config = yaml.safe_load(f)

with open('./prompts/evaluate_prompt.yml', 'r') as f:
    evaluate_prompt_template = yaml.safe_load(f)

llm = LargeLanguageModel(**model_config)
llm.load_prefix_suffix(prefix_text=prompt_config['prefix'], suffix_text=prompt_config['suffix'])
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

mongo = MongoDBController(
    host=os.environ['mongodb_url'],
    port=int(os.environ['mongodb_port']), 
    username=os.environ['mongodb_user'], 
    password=os.environ['mongodb_password'],
    db_name='test_data'
)

writer = pd.ExcelWriter(args.save_path)
for dataset_name, df_question in df_questions.items():
    df = mongo.find_all(collection_name=dataset_name)
    
    dataframe_agent = create_pandas_dataframe_agent(
        llm.llm, 
        df,
        verbose=True,
        prefix=prefix,
        suffix=suffix,
        input_variables=['input', 'agent_scratchpad', 'df_head'],
        agent_executor_kwargs={'handle_parsing_errors': True},
        include_df_in_prompt=include_df_in_prompt,
        return_intermediate_steps=True,
        max_iterations=5,
        max_execution_time=600,
        early_stopping_method='force', 
        number_of_head_rows=5
    )
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


# Load the Excel file
with pd.ExcelFile(args.save_path) as xls:
    # Get the names of all sheets in the Excel file
    sheet_names = xls.sheet_names

# Initialize a dictionary to store accuracy for each dataset
accuracy_per_dataset = {}

# Initialize counters for overall correct predictions and total predictions
overall_correct_predictions = 0
overall_total_predictions = 0

# For each dataset (sheet in the Excel file)
for dataset_name in sheet_names:
    # Read the dataset into a DataFrame
    df = pd.read_excel(args.save_path, sheet_name=dataset_name)

    # Calculate the number of correct predictions (rating == 1) and total number of predictions
    correct_predictions = (df['rating'] == 1).sum()
    total_predictions = len(df)
    
    # Update the overall counters
    overall_correct_predictions += correct_predictions
    overall_total_predictions += total_predictions
    
    # Calculate the accuracy for the current dataset
    accuracy = correct_predictions / total_predictions
    
    # Store the accuracy in the dictionary
    accuracy_per_dataset[dataset_name] = accuracy

# Calculate the overall accuracy
overall_accuracy = overall_correct_predictions / overall_total_predictions

# Print the accuracy for each dataset
# Print the accuracy for each dataset
for dataset_name, accuracy in accuracy_per_dataset.items():
    print(f'Accuracy for {dataset_name}: {accuracy * 100:.2f}%')

# Print the overall accuracy
print(f'Overall accuracy: {overall_accuracy * 100:.2f}%')