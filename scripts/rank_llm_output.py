import sys
sys.path.append('.')

from utils.llm import LargeLanguageModel
from utils.common import read_yaml
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

import argparse

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--results', type=str, help='results csv', required=True)
ap.add_argument('--output', type=str, help='output path', required=True)
args = ap.parse_args()

model_config = read_yaml('./model_configs/openchat.yml')
llm = LargeLanguageModel(**model_config)
eval_prompt_template = read_yaml('./prompts/evaluate_prompt.yml')

df_result = pd.read_csv(args.results)

df_eval = []
prompt_template = PromptTemplate.from_template(model_config['prompt_template'])
runnable = prompt_template | llm.llm | StrOutputParser()
for idx, row in df_result.iterrows():
    evaluation_prompt = eval_prompt_template['user'].format(
        query=row['question'],
        ground_truth=row['ground_truth'],
        answer=row['llm_output']
    )
    rating = runnable.invoke({'system_message': eval_prompt_template['system'], 'prompt': evaluation_prompt})
    row['rating'] = int(rating)
    df_eval.append(row)

df_eval = pd.DataFrame(df_eval)
df_eval.loc[0, 'accuracy'] = df_eval['rating'].mean()

df_eval.to_csv(args.output, index=False)