from langchain_community.llms import LlamaCpp, Ollama,VLLM
from langchain_openai import OpenAI, ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

from aserious_agent.pandas_agent import create_pandas_dataframe_agent
import yaml

class LargeLanguageModelAgent:
    def __init__(self, model_config_path):
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)

        self.llm = LargeLanguageModel(**model_config)

    def create_dataframe_agent(self, df, desc, prompt_example):
        return create_pandas_dataframe_agent(
            self.llm.llm, 
            df,
            table_desc=desc,
            prompt_example=prompt_example,
            verbose=True,
            prefix=self.llm.prefix,
            suffix=self.llm.suffix,
            input_variables=['input', 'agent_scratchpad', 'df_head', 'table_desc', 'date_time', 'prompt_example'],
            agent_executor_kwargs={'handle_parsing_errors': True},
            include_df_in_prompt=True,
            return_intermediate_steps=True,
            max_iterations=6,
            max_execution_time=600,
            early_stopping_method='generate'
        )

class LargeLanguageModel:
    def __init__(
        self, **kwargs
    ):
        if (kwargs['model_type'] == 'llama-cpp'):
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            self.llm = LlamaCpp(
                model_path=kwargs['model_path'],
                temperature=kwargs['temperature'],
                max_tokens=kwargs['max_tokens'],
                top_p=kwargs['top_p'],
                callback_manager=callback_manager,
                verbose=True,
                streaming=True,
                # stop=kwargs['stop'],
                n_gpu_layers=kwargs['n_gpu_layers'],
                n_ctx=kwargs['context_length'],
                offload_kqv=True,
                seed=6969
            )   
    
        elif (kwargs['model_type'] == 'ollama'):
            self.llm = Ollama(
                model=kwargs['model'],
                temperature=kwargs['temperature'],
                repeat_last_n=-1,
                num_ctx=kwargs['context_length'],
            )
        elif (kwargs['model_type'] == 'vllm'):
            dtype = "half" if kwargs['quantization'] == "gptq" else "auto"
            self.llm = VLLM(
                model=kwargs['hfmodel'],
                tensor_parallel_size=kwargs['gpu_size'],
                trust_remote_code=True,  # mandatory for hf models
                max_new_tokens=kwargs['max_new_tokens'],
                top_k=kwargs['top_k'],
                top_p=kwargs['top_p'],
                temperature=kwargs['temperature'],
                vllm_kwargs={"quantization": kwargs['quantization']},
                gpu_memory_utilization=1.0, #can adjust your gpu usage to 0.8,0.5 ....
                download_dir=kwargs['download_dir'],
                dtype=dtype
            )

        else:
            import os
            from dotenv import load_dotenv
            load_dotenv('.env')
            if 'instruct' in kwargs['gpt_type']:
                self.llm = OpenAI(model_name=kwargs['gpt_type'], openai_api_key=os.environ['openai_api_key'])
            else:
                self.llm = ChatOpenAI(model_name=kwargs['gpt_type'], openai_api_key=os.environ['openai_api_key'])

        # simple runnable
        self.prompt_template = PromptTemplate.from_template(kwargs['prompt_template'])
        self.llm_runnable = self.prompt_template | self.llm | StrOutputParser()

        if kwargs.get('prefix_template'):
            self.prefix_template = kwargs['prefix_template']
            self.suffix_template = kwargs['suffix_template']
    
    def load_prefix_suffix(self, prefix_text, suffix_text):
        self.prefix = self.prefix_template.format(prefix_text=prefix_text)
        self.suffix = self.suffix_template.format(suffix_text=suffix_text)

        return self.prefix, self.suffix