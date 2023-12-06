from llm import LargeLanguageModel
from aserious_agent.pandas_agent import create_pandas_dataframe_agent
from langchain.globals import set_verbose
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

import os
import sys
import yaml
import time
import argparse
import traceback

import numpy as np
import pandas as pd
import wandb

set_verbose(True)

ap = argparse.ArgumentParser()
ap.add_argument('--save_path', type=str, required=True)
ap.add_argument('--model', type=str, default='./model_configs/neural-chat.yml')
ap.add_argument('--prompt', type=str, default='./prompts/pandas_prompt_01.yml')
ap.add_argument('--use-custom-prompt', action='store_true', default=False)
ap.add_argument('--questions_answers', type=str, default='./data/questions_answers.xlsx')
args = ap.parse_args()

save_folder = args.save_path.split('/')
save_folder = '/'.join(save_folder[:-1])
os.makedirs(save_folder, exist_ok=True)

os.environ['LANGCHAIN_WANDB_TRACING'] = 'true'
os.environ['WANB_PROJECT'] = 'tmp-llm'

with open(args.model, 'r') as f:
    model_config = yaml.safe_load(f)

with open(args.prompt, 'r') as f:
    prompt_config = yaml.safe_load(f)

with open('./prompts/evaluate_prompt.yml', 'r') as f:
    evaluate_prompt_template = yaml.safe_load(f)

llm = LargeLanguageModel(**model_config, **prompt_config)

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
    df = pd.read_csv(f'data/csv_data/{dataset_name}.csv')
    
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
        max_iterations=10,
        max_execution_time=600,
        early_stopping_method='force', 
    )
    results = []
    with wandb.init(project='langchain-tracing') as run:
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
                runnable = evaluation_prompt | llm.llm | StrOutputParser()
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
