import yaml
import os
import re
import json

import sentencepiece as spm

import pandas as pd

from pydantic import BaseModel

ENV = os.environ['SUDUAI_ENV']

def read_yaml(file_path):
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)

    return data

def tokenize(tokenizer_file, text):
    sp = spm.SentencePieceProcessor(model_file=tokenizer_file)
    tokens = sp.encode(text)

    return tokens

def convert_to_date(df, date_pattern=None):
    def convert_column_to_date(date_series):
        try:
            return pd.to_datetime(date_series, dayfirst=True)
        except ValueError:
            return date_series
    
    if date_pattern is None:
        date_pattern = r'(\b\d{4}-\d{2}-\d{2}\b)|(\b\d{1,2}/\d{1,2}/\d{2,4}\b)|(\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s\d{1,2},?\s\d{2,4}\b)|(\b(?:Mon(?:day)?|Tue(?:sday)?|Wed(?:nesday)?|Thu(?:rsday)?|Fri(?:day)?|Sat(?:urday)?|Sun(?:day)?)\s(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s\d{1,2},?\s\d{2,4}\b)|(\b\d{2}-\d{2}-\d{4}\b)'
    
    date_columns = []
    for column in df.columns:
        if df[column].dtype == 'object' or df[column].dtype == "datetime64[ns]" or df[column].dtype == "str":
            date_matches = df[column].astype(str).str.match(date_pattern, na=False)
            if date_matches.any(): 
                df[column] = convert_column_to_date(df[column])
                date_columns.append(column)
    
    if not date_columns:
        return df
    
    for date_col in date_columns:
        df[f"{date_col}_Year"] = df[date_col].dt.year
        df[f"{date_col}_Month"] = df[date_col].dt.month
        df[f"{date_col}_Day"] = df[date_col].dt.day
    
    return df.drop(columns=date_columns)

def parse_langchain_debug_log(debug_log):
    try:
        pattern = r'Entering LLM run with input:\n\[0m(.*?){(.*?)}(.*?)\[36;1m\[1;3m\[llm/end\]'
        matches = re.findall(pattern, debug_log, re.DOTALL)
        pre_log = json.loads('{' + matches[-1][1] + '}')['prompts'][0]
        print(pre_log)

        # final answer
        pattern = r'ReActSingleInputOutputParser] Entering Parser run with input:\n\[0m(.*?){(.*?)}(.*?)\[36;1m\[1;3m\[chain/end\]'
        matches = re.findall(pattern, debug_log, re.DOTALL)
        post_log = json.loads('{' + matches[-1][1] + '}')['input']
        final_log = pre_log + post_log
    except:
        final_log = debug_log

    return final_log

class LogData(BaseModel):
    class Config:
        extra = 'allow'
