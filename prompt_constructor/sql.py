from prompt_constructor.base import BasePromptConstructor
from utils.llm import LargeLanguageModel
from langchain_core.prompts import PromptTemplate

import datetime
import pytz

class SqlPromptConstructor(BasePromptConstructor):
    def __init__(self, llm: LargeLanguageModel, system_message: str, user_message: str):
        super().__init__(llm, system_message, user_message)
        partial_prompt = self.prompt_template.format(
            system_message=self.system_message,
            user_message=self.user_message
        )
        self.partial_prompt = PromptTemplate.from_template(partial_prompt)

    def get_prompt(self, table_name: str, table_schema_markdown: str, retrieval_description: str, prompt_example: str):
        
        now = datetime.datetime.now(pytz.timezone('Asia/Singapore'))
        date_now = now.strftime('%Y-%m-%d')

        sql_prompt = self.partial_prompt.partial(
            table_name=table_name,
            table_schema_markdown=table_schema_markdown,
            retrieval_description=retrieval_description,
            prompt_example=prompt_example,
            date_now=date_now
        )

        return sql_prompt
