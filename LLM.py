# LargeLanguageModelAgent.py
from llm import LargeLanguageModel
from aserious_agent.pandas_agent import create_pandas_dataframe_agent
import yaml

class LargeLanguageModelAgent:
    def __init__(self, model_config_path, prompt_config_path):
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)

        with open(prompt_config_path, 'r') as f:
            prompt_config = yaml.safe_load(f)

        self.llm = LargeLanguageModel(**model_config, **prompt_config)

    def create_dataframe_agent(self, df):
        return create_pandas_dataframe_agent(
            self.llm.llm, 
            df,
            verbose=True,
            prefix=self.llm.prefix,
            suffix=self.llm.suffix,
            input_variables=['input', 'agent_scratchpad', 'df_head'],
            agent_executor_kwargs={'handle_parsing_errors': True},
            include_df_in_prompt=True,
            return_intermediate_steps=True,
            max_iterations=10,
            max_execution_time=600,
            early_stopping_method='force', 
        )
