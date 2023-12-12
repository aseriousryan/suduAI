# Docker setup
Create container network
```
$ docker network create sudu
```

Create Ollama container to host models
```
$ docker run -d --gpus=all -v <local models directory>:/root/.ollama -p 11434:11434 --network sudu --name ollama ollama/ollama
```