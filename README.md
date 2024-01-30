# Docker setup / Release flow
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

# Update Table Description Embedding in MongoDB After Training
```
$ python scripts/update_desc_emb.py --env development --model models/Trained_Model_V5/
```