import yaml

import sentencepiece as spm

def read_yaml(file_path):
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)

    return data

def tokenize(tokenizer_file, text):
    sp = spm.SentencePieceProcessor(model_file=tokenizer_file)
    tokens = sp.encode(text)

    return tokens