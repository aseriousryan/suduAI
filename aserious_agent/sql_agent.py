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
from langchain.sql_database import SQLDatabase
import pandas as pd
from utils.mongoDB import MongoDBController
from utils.collection_retriever import sentence_transformer_retriever
from utils.prompt_retriever import prompt_example_sentence_transformer_retriever
from utils.row_retriever import top5_row_for_question

import time
import os

class SQLAgent:
    def __init__(
        self,
        llm: LargeLanguageModel,
        data_logger: LogData,
        mongo: MongoDBController,
        db: SQLDatabase,
        max_iterations: int = 8,
    ):
        
        self.llm = llm
        self.max_iterations = max_iterations
        self.data_logger = data_logger
        self.db = db
        self.mongo = mongo

        # construct prompt
        self.system_message = read_yaml(os.environ.get('prompt'))['system_message']
        self.user_message = read_yaml(os.environ.get('prompt'))['user_message']
        self.prompt_constructor = SqlPromptConstructor(llm, self.system_message, self.user_message)
    
    def run_agent(self, user_query: str,database_name: str, collection: str = None):

        # retrieve collection
        start = time.time()
        if collection is None:
            collection, table_desc, desc_cos_sim = sentence_transformer_retriever(user_query, database_name)
        else:
            table_desc = self.mongo.get_table_desc(database_name, collection)
            desc_cos_sim = -1

        df_data = self.mongo.find_all(database_name, collection, exclusion={'_id': 0})
        if df_data.shape[0] == 0:
            raise RuntimeError(f'No data found:\ndb: {database_name}\ncollection: {collection}')

        # retrieve prompt example
        prompt_example, question_retrieval = prompt_example_sentence_transformer_retriever(user_query, database_name)

        # retrieve top 5 rows
        df_top5 = str(top5_row_for_question(user_query, df_data).to_markdown())

        end = time.time()
        retrieval_time = end - start

        # toolkit = SQLDatabaseToolkit(llm=self.llm.llm, db=self.db).get_tools()

        self.tools = []
        self.prompt = self.prompt_constructor.get_prompt(prompt_example, table_desc, df_top5)
        self.create_agent()

        start = time.time()
        result = self.agent_executor.invoke({'input': user_query})
        end = time.time()
        response_time = end - start

         # log data
        self.data_logger.collection = collection
        self.data_logger.desc_cos_sim = desc_cos_sim
        self.data_logger.query_retrieval = question_retrieval
        self.data_logger.collection_retrieval_time = retrieval_time
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


