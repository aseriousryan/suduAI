import yaml

import sentencepiece as spm

import pandas as pd

def read_yaml(file_path):
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)

    return data

def tokenize(tokenizer_file, text):
    sp = spm.SentencePieceProcessor(model_file=tokenizer_file)
    tokens = sp.encode(text)

    return tokens

def convert_to_date(date_str):
    try:
        date_conversion = pd.to_datetime(date_str,  dayfirst=False)
    
    except ValueError:
        date_conversion =  date_str  
    
    return date_conversion
