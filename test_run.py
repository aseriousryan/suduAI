from llm import LargeLanguageModel
from sentence_transformers import SentenceTransformer, util
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.load.dump import dumps
from langchain.globals import set_verbose
import os
import sys
import yaml
import time
import argparse
import traceback
import numpy as np
import pandas as pd
import wandb

# prefix= """### System:
# You are working with a pandas dataframe in Python. The name of the dataframe is "df".
# You should use the tools below to answer the question posed of you: """

prefix= """### System:You are an agent designed to interact with a csv table in a pandas dataframe by Python. The name of the dataframe is "df".
Given an input question, create a syntactically correct pandas dataframe operations to run, then look at the results of the pandas operations and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your pandas operations to at most 10 results.
You can order the results by a relevant column to return the most interesting examples in the table.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the table.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your python pandas operations before executing it. If you get an error while executing a python pandas operations, rewrite the pandas operations and try again.

If the question does not seem related to the table, just return "I don't know" as the answer."""

suffix = """Begin! 
### User:
Question: {input}
Thought: I should look at the available tables to see what I can index from.  Then I should perform python pandas operations on the most relevant tables.
{agent_scratchpad} 
### Assistant:"""

# suffix = """Begin! 
# ### User:
# Question: {input}
# {agent_scratchpad} 
# ### Assistant:"""



# prefix= """<s>[INST]
# You are working with a pandas dataframe in Python. The name of the dataframe is "df".
# You should use the tools below to answer the question posed of you:

# python_repl_ast: A Python shell. Use this to execute python commands. Input should be a valid python command. 
# When using this tool, sometimes output is abbreviated - make sure it does not look abbreviated before using it in your answer."""

# suffix = """Begin! 
# Question: {input}
# {agent_scratchpad} [/INST] 
# """

set_verbose(True)

log_file_path = '/home/seriousco/Documents/dennis/testing2/Log/logfile.txt'
if os.path.exists(log_file_path):
    os.remove(log_file_path)
sys.stdout = open(log_file_path, 'w')
sys.stderr = open(log_file_path, 'a')

ap = argparse.ArgumentParser()
ap.add_argument('--model')
ap.add_argument('--save_path')
args = ap.parse_args()

save_folder = args.save_path.split('/')
save_folder = '/'.join(save_folder[:-1])
os.makedirs(save_folder, exist_ok=True)

os.environ['LANGCHAIN_WANDB_TRACING'] = 'true'
os.environ['WANB_PROJECT'] = 'tmp-llm'

scoring_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

with open(args.model, 'r') as f:
    model_config = yaml.safe_load(f)

llm = LargeLanguageModel(**model_config)

df_questions = pd.read_excel('./data/questions_answers.xlsx',sheet_name=None)

writer = pd.ExcelWriter(args.save_path)
for dataset_name, df_question in df_questions.items():
    df = pd.read_csv(f'data/csv_data/{dataset_name}.csv')
    
    dataframe_agent = create_pandas_dataframe_agent(
        llm.llm, 
        df,
        verbose=True,
        # prefix=prefix,
        # suffix=suffix,
        # input_variables=["input","agent_scratchpad" ],
        agent_executor_kwargs={"handle_parsing_errors": True},
        include_df_in_prompt=None,
        return_intermediate_steps=True,
        max_iterations=10,
        max_execution_time=12,
        early_stopping_method="force", 
    )

    results = []
    with wandb.init(project="langchain-tracing") as run:
        for idx, row in df_question.iterrows():
            question = row['Questions']
            answer = str(row['Answers'])
            start = time.time()
            last_key_value_pairs = []  # Initialize an empty list to collect last key-value pairs
            try:
                result = dataframe_agent({'input': question})
                llm_output = str(result['output'])
                similarity = util.pytorch_cos_sim(
                    scoring_model.encode([answer], convert_to_tensor=True),
                    scoring_model.encode(llm_output, convert_to_tensor=True)
                )[0].item()

                intermediate_steps = result["intermediate_steps"]
                for step in intermediate_steps:
                    print(step)
                    last_key_value_pair = str(step).split(",")[-1]  # Extract the last key-value pair
                    last_key_value_pairs.append(last_key_value_pair)
                    
                
            except:
                llm_output = traceback.format_exc()
                similarity = -1
            end = time.time()

            results.append({
                'question': question,
                'execution_time': np.round(end-start, 2),
                'llm_output': llm_output,
                'answer': answer,
                'score': similarity,
                'last_key_value_pairs': last_key_value_pairs  # Add last key-value pairs to the dictionary
            })

    df_result = pd.DataFrame(results)
    df_result.to_excel(writer, index=False, sheet_name=dataset_name)

writer.close()
