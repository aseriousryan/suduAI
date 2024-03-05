## Docker setup / Release flow
Build images
```
$ python scripts/build_docker.py --model_path <model path> --env <production | development>
```

To run chat endpoint image
```
$ docker run --gpus all -p 8080:8080 asai-sudu:0.1
```

To run upload endpoint image
```
$ docker run -d -p 8082:8082 --name upload_sudu registry.gitlab.com/dark_knight/aserious-sudu:upload-0.1
```

## Update Table Description Embedding in MongoDB After Training
```
$ python scripts/update_desc_emb.py --env development --model models/Trained_Model_V5/
```

## Backup MongoDB (download MongoDB to a local folder)
```
$ python scripts/backup_mongodb.py --output tmp/ --env <production|development>
```

## Evaluate LLM Output using LLM
```
$ python scripts/rank_llm_output.py --results tmp/results.csv --output tmp/ranked.csv
```
Please note that `results.csv` must have the following 3 columns: question, ground_truth and llm_output

## Automation in question generation
```
$ python question_scripts/script1.py 
```
script1 is for running through an excel sheet with questions for LLM answer.

## Uploading the Prompt Example
Prompt example have 3 columns: question, log, collection_name
```
$ unicorn prompt_example_upload:app --port 8085
```
After running the port, go to port on web browser to upload the prompt example excel file in the database.
