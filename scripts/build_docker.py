import sys
sys.path.append('.')

import subprocess
import os
import shutil
import argparse

from dotenv import load_dotenv
from utils.common import read_yaml

ap = argparse.ArgumentParser()
ap.add_argument('--model_path', type=str, required=True, help='Actual directory to model weights')
ap.add_argument('--env', type=str, default='development', help='production | development')
ap.add_argument('--build', nargs='+', default=['chat', 'upload'])
args = ap.parse_args()

load_dotenv(f'./.env.{args.env}')

model_config = read_yaml(os.environ['model'])
model = os.path.basename(model_config['model_path'])
sent_trans_model = os.path.normpath(os.path.basename(model_config['collection_retriever_sentence_transformer']))
tokenizer = os.path.basename(os.environ['tokenizer'])
prompt = os.path.basename(os.environ['prompt'])
version = open('version.md').read()

# copy weights over to current folder for docker build
if not os.path.exists(model):
    print('[*] Copying weights to project root...')
    shutil.copy(os.path.join(args.model_path, model), model)
if not os.path.exists(tokenizer):
    print('[*] Copying tokenizer to project root...')
    shutil.copy(os.path.join(args.model_path, tokenizer), tokenizer)
if not os.path.isdir(sent_trans_model):
    print('[*] Copying collection retriever sentence transformer model to project root...')
    shutil.copytree(os.path.join(args.model_path, sent_trans_model), sent_trans_model)

if 'chat' in args.build:
    build_cmd = f'docker build -f Dockerfile --no-cache ' \
        f'--build-arg MODEL={model} --build-arg TOKENIZER={tokenizer} ' \
        f'--build-arg PROMPT={prompt} --build-arg SENT_TRANS_MODEL={sent_trans_model} ' \
        f'--build-arg SUDUAI_ENV={args.env} ' \
        f'-t asai-sudu:{version} .'
    print(f'[*] Building chat app Docker image:\n{build_cmd}\n')
    subprocess.run(build_cmd, shell=True, text=True)

if 'upload' in args.build:
    build_cmd = f'docker build -f Dockerfile_upload --no-cache ' \
        f'--build-arg TOKENIZER={tokenizer} ' \
        f'--build-arg SUDUAI_ENV={args.env} ' \
        f'-t asai-sudu:upload-{version} .'
    print(f'[*] Building upload app Docker image:\n{build_cmd}\n')
    subprocess.run(build_cmd, shell=True, text=True)

# clean up
os.remove(model)
os.remove(tokenizer)

