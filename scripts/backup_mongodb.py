import sys
sys.path.append('.')

from dotenv import load_dotenv
from datetime import datetime

import subprocess
import argparse
import os

parser = argparse.ArgumentParser(description='Backup MongoDB collections.')
parser.add_argument('--output', required=False, default=None, help='MongoDB connection URL')
parser.add_argument('--env', default='production', help='Data environment to backup')
args = parser.parse_args()

load_dotenv(f'./.env.{args.env}')

def backup_mongodb(backup_path=None):
    output_folder = f'{datetime.now().strftime("%Y%m%d_%H%M")}-{args.env}-mongobackup'
    if backup_path:
        output_folder = os.path.join(backup_path, output_folder)

    command = f'./scripts/mongodb-database-tools-ubuntu2204-x86_64-100.9.4/bin/mongodump ' \
        f'--uri mongodb://{os.environ["mongodb_user"]}:{os.environ["mongodb_password"]}@{os.environ["mongodb_url"]}:{os.environ["mongodb_port"]}/ ' \
        f'--authenticationDatabase=admin --out {output_folder}'
    
    try:
        subprocess.run(command, check=True, shell=True)
        print(f'Backup completed successfully')
    except subprocess.CalledProcessError as e:
        print(f'Error during backup: {e}')

if __name__ == '__main__':
    backup_mongodb(args.output)
