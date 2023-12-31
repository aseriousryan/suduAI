FROM ubuntu:22.04

ARG MODEL
ARG TOKENIZER
ARG SUDUAI_ENV

RUN apt-get update && apt-get -y install python3 python3-pip pkg-config python3-dev git htop build-essential nvidia-cuda-toolkit

# SET TIMEZONE TO AVOID ANSWERING PROMPTS
RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime
RUN echo Etc/UTC > /etc/timezone

WORKDIR /app
COPY requirements.txt .env.$SUDUAI_ENV app.py version.md  .
COPY utils utils/
COPY aserious_agent aserious_agent/
COPY model_configs model_configs/
RUN mkdir prompts/
COPY prompts/collection_retriever_prompt.yml prompts/collection_retriever_prompt.yml

# model weights
RUN mkdir models/
COPY $MODEL models/$MODEL
COPY $TOKENIZER models/$TOKENIZER

RUN pip install -r requirements.txt
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install --force-reinstall llama-cpp-python --no-cache-dir

ENV SUDUAI_ENV=$SUDUAI_ENV
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]