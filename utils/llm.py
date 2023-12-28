from langchain.llms import LlamaCpp, OpenAI, Ollama
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

    def create_dataframe_agent(self, df, desc):
        return create_pandas_dataframe_agent(
            self.llm.llm, 
            df,
            table_desc=desc,
            verbose=True,
            prefix=self.llm.prefix,
            suffix=self.llm.suffix,
            input_variables=['input', 'agent_scratchpad', 'df_head', 'table_desc'],
            agent_executor_kwargs={'handle_parsing_errors': True},
            include_df_in_prompt=True,
            return_intermediate_steps=True,
            max_iterations=10,
            max_execution_time=600,
            early_stopping_method='force', 
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
                top_p=1,
                callback_manager=callback_manager,
                verbose=True,
                streaming=True,
                # stop=kwargs['stop'],
                n_gpu_layers=kwargs['n_gpu_layers'],
                n_ctx=kwargs['context_length']
            )   
    
        elif (kwargs['model_type'] == 'ollama'):
            self.llm = Ollama(
                model=kwargs['model'],
                temperature=kwargs['temperature'],
                repeat_last_n=-1,
                num_ctx=kwargs['context_length'],
            )
        else:
            import os
            from dotenv import load_dotenv
            load_dotenv('.env')
            self.llm = OpenAI(openai_api_key=os.environ['openai_api_key'])

        # simple runnable
        self.prompt_template = PromptTemplate.from_template(kwargs['prompt_template'])
        self.llm_runnable = self.prompt_template | self.llm | StrOutputParser()

        self.prefix_template = kwargs['prefix_template']
        self.suffix_template = kwargs['suffix_template']
    
    def load_prefix_suffix(self, prefix_text, suffix_text):
        self.prefix = self.prefix_template.format(prefix_text=prefix_text)
        self.suffix = self.suffix_template.format(suffix_text=suffix_text)

        return self.prefix, self.suffix