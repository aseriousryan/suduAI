FROM nvidia/cuda:11.8.0-base-ubuntu22.04

RUN apt-get update && apt-get -y install python3 python3-pip git htop build-essential

# SET TIMEZONE TO AVOID ANSWERING PROMPTS
RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime
RUN echo Etc/UTC > /etc/timezone

WORKDIR ./
COPY requirements.txt llm.py aserious_agent model_configs prompts ./
RUN pip install -r requirements.txt
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

