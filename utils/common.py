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

def parse_langchain_debug_log(debug_log): #json
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



def extract_prompts(input_text):
    # Define the pattern to match all occurrences of the "prompts" array content
    prompts_pattern = re.compile(r'"prompts": \[([\s\S]*?)\]', re.DOTALL)

    # Define the pattern to match the last occurrence of the "log" value
    log_pattern = re.compile(r'"log": "(.*?)"', re.DOTALL)

    # Find all occurrences of "prompts" in the input text
    prompts_matches = prompts_pattern.findall(input_text)

    # Check if there are any "prompts" matches
    if prompts_matches:
        # Extract the content of the last "prompts" array
        last_prompts = prompts_matches[-1]

        # Define the pattern to match the last occurrence of the "log" value
        log_pattern = re.compile(r'"log": "(.*?)"', re.DOTALL)

        # Search for the "log" pattern in the input text
        log_match = log_pattern.findall(input_text)

        # Check if a "log" match is found
        if log_match:
            # Extract the last occurrence of the "log" value
            last_log = log_match[-1]

            # Append the last log to the extracted prompts
            last_prompts += ']\nThought: ' + last_log

        # Replace "\n" with actual newline character
        last_prompts = last_prompts.replace("\\n", "\n")

        return last_prompts  

    return "Prompts not found in the input text."


class LogData(BaseModel):
    class Config:
        extra = 'allow'


def is_date_column(series):
    # Check for common date formats
    common_formats = [
        '%Y-%m-%d', '%Y/%m/%d', '%Y%m%d',
        '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%Y%m%d %H:%M:%S',
        '%Y-%m-%d %H:%M', '%Y/%m/%d %H:%M', '%Y%m%d %H:%M',
        '%d-%m-%Y', '%d/%m/%Y', '%d%m%Y',
        '%d-%m-%Y %H:%M:%S', '%d/%m/%Y %H:%M:%S', '%d%m%Y %H:%M:%S',
        '%d-%m-%Y %H:%M', '%d/%m/%Y %H:%M', '%d%m%Y %H:%M'
    ]
    
    for format_str in common_formats:
        try:
            pd.to_datetime(series, format=format_str, errors='raise')
            return True
        except (ValueError, pd.errors.ParserError):
            pass

    return False

def convert_date_columns_to_sql_datetime(df):
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            # If column is already datetime-like, do nothing
            continue
        elif pd.api.types.is_string_dtype(df[col]) and is_date_column(df[col]):
            try:
                df[col] = pd.to_datetime(df[col], errors='raise')
                print(type(df[col]))
            except (pd.errors.ParserError, pd.errors.OutOfBoundsDatetime):
                # If parsing or out of bounds fails, continue checking other columns
                continue

    return df

def generate_sql_table_schema_markdown(table_name, columns_str, data_types_str):
    # Split the input strings into lists
    columns = [col.strip() for col in columns_str.split(';')]
    data_types = [dtype.strip() for dtype in data_types_str.split(';')]

    # Check if the number of columns and data types match
    if len(columns) != len(data_types):
        raise ValueError("Number of columns and data types must be the same.")

    # Create the SQL table schema markdown
    table_schema_markdown = f"## Table name: {table_name}\n\n"
    table_schema_markdown += "| Column Name | Data Type |\n"
    table_schema_markdown += "|--------------|------------|\n"

    for col, dtype in zip(columns, data_types):
        # Add double quotes around the column name
        col_with_quotes = f'"{col}"'
        table_schema_markdown += f"| {col_with_quotes} | {dtype} |\n"

    return table_schema_markdown



