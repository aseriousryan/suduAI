import warnings
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from langchain.agents import create_react_agent
from langchain.agents.agent import (
    AgentExecutor,
    BaseMultiActionAgent,
    BaseSingleActionAgent,
    RunnableAgent,
    RunnableMultiActionAgent,
)

from langchain_core.language_models import LanguageModelLike
from langchain_core.tools import BaseTool
from langchain_core.utils.interactive_env import is_interactive_env

from langchain_core.callbacks import BaseCallbackManager
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.tracers import ConsoleCallbackHandler

from langchain_experimental.tools.python.tool import PythonAstREPLTool

from prompt_constructor.pandas import PandasPromptConstructor
from utils.llm import LargeLanguageModel
from utils.common import read_yaml

import pandas as pd

import os

class PandasAgent:
    def __init__(
        self,
        llm: LargeLanguageModel, 
        df: pd.DataFrame,
        prompt_example: str,
        table_desc: str,
        df_head: str,
        max_iterations: int = 7,
    ):
        self.tools = [PythonAstREPLTool(locals={'df': df})]
        self.llm = llm
        self.max_iterations = max_iterations

        # construct prompt
        self.system_prompt = read_yaml(os.environ.get('prompt'))['prompt']
        self.prompt_constructor = PandasPromptConstructor(llm, self.system_prompt)
        self.prompt = self.prompt_constructor.get_prompt(prompt_example, table_desc, df_head)
    
    def create_agent(self):
        self.agent: Union[BaseSingleActionAgent, BaseMultiActionAgent] = RunnableAgent(
            runnable=create_react_agent(self.llm.llm, self.tools, self.prompt),
            input_keys_arg=["input"],
            return_keys_arg=["output"],
        )

        return AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            callback_manager=CallbackManager([ConsoleCallbackHandler()]),
            verbose=False,
            return_intermediate_steps=False,
            max_iterations=self.max_iterations,
            max_execution_time=600,
            early_stopping_method='force',
            handle_parsing_errors=True,
            **{},
        )