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
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.tracers import ConsoleCallbackHandler

from utils.llm import LargeLanguageModel
from utils.common import read_yaml, LogData
from utils.row_retriever import top5_row_for_question
from prompt_constructor.sql import SqlPromptConstructor
from tools.sql_database_toolkit import SQLDatabaseToolkit
from tools.sql_tool import SQLQueryTool
from langchain.sql_database import SQLDatabase
import pandas as pd
from utils.mongoDB import MongoDBController
from utils.collection_retriever import sentence_transformer_retriever
from utils.prompt_retriever import prompt_example_sentence_transformer_retriever
from utils.row_retriever import top5_row_for_question
from utils.table_retriever import table_schema_retriever
from sqlalchemy.engine import Engine

import time
import os

class SQLAgent:
    def __init__(
        self,
        llm: LargeLanguageModel,
        data_logger: LogData,
        mongo: MongoDBController,
        engine: Any,
        max_iterations: int = 8,
    ):
        
        self.llm = llm
        self.max_iterations = max_iterations
        self.data_logger = data_logger
        self.engine = engine
        self.mongo = mongo

        # construct prompt
        self.system_message = read_yaml(os.environ.get('prompt'))['system_message']
        self.user_message = read_yaml(os.environ.get('prompt'))['user_message']
        self.prompt_constructor = SqlPromptConstructor(llm, self.system_message, self.user_message)
    
    def run_agent(self, user_query: str,database_name: str, collection: str = None):

        # retrieve table schema
        start = time.time()

        if collection is None:
            table_name, table_schema_markdown, retrieval_description, desc_cos_sim  = table_schema_retriever(user_query, database_name)
       
        end = time.time()
        retrieval_time = end - start

        self.tools = [SQLQueryTool(self.engine)]
        self.prompt = self.prompt_constructor.get_prompt(table_name, table_schema_markdown, retrieval_description)
        self.create_agent()

        start = time.time()
        result = self.agent_executor.invoke({'input': user_query})
        end = time.time()
        response_time = end - start

        # log data
        self.data_logger.table = table_name
        self.data_logger.retrieval_desc = retrieval_description
        self.data_logger.desc_cos_sim = desc_cos_sim
        self.data_logger.table_retrieval_time = retrieval_time
        self.data_logger.response_time = response_time

        return result

    def create_agent(self):
        self.agent: Union[BaseSingleActionAgent, BaseMultiActionAgent] = RunnableAgent(
            runnable=create_react_agent(self.llm.llm, self.tools, self.prompt),
            input_keys_arg=["input"],
            return_keys_arg=["output"],
        )

        self.agent_executor = AgentExecutor(
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


