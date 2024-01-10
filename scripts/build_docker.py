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
args = ap.parse_args()

load_dotenv(f'./.env.{args.env}')

model_config = read_yaml(os.environ['model'])
model = os.path.basename(model_config['model_path'])
tokenizer = os.path.basename(os.environ['tokenizer'])
version = open('version.md').read()

# copy weights over to current folder for docker build
if not os.path.exists(model):
    print('[*] Copying weights to project root...')
    shutil.copy(os.path.join(args.model_path, model), model)
if not os.path.exists(tokenizer):
    print('[*] Copying tokenizer to project root...')
    shutil.copy(os.path.join(args.model_path, tokenizer), tokenizer)

build_cmd = f'docker build -f Dockerfile --no-cache ' \
    f'--build-arg MODEL={model} --build-arg TOKENIZER={tokenizer} ' \
    f'--build-arg SUDUAI_ENV={args.env} ' \
    f'-t asai-sudu:{version} .'
print(f'[*] Building chat app Docker image:\n{build_cmd}\n')
subprocess.run(build_cmd, shell=True, text=True)

build_cmd = f'docker build -f Dockerfile_upload --no-cache ' \
    f'--build-arg TOKENIZER={tokenizer} ' \
    f'--build-arg SUDUAI_ENV={args.env} ' \
    f'-t asai-sudu:upload-{version} .'
print(f'[*] Building upload app Docker image:\n{build_cmd}\n')
subprocess.run(build_cmd, shell=True, text=True)

# clean up
os.remove(model)
os.remove(tokenizer)

