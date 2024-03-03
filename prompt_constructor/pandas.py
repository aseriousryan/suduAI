from prompt_constructor.base import BasePromptConstructor
from utils.llm import LargeLanguageModel
from langchain_core.prompts import PromptTemplate

import datetime
import pytz

class PandasPromptConstructor(BasePromptConstructor):
    def __init__(self, llm: LargeLanguageModel, system_message: str, user_message: str):
        super().__init__(llm, system_message, user_message)
        partial_prompt = self.prompt_template.format(
            system_message=self.system_message,
            user_message=self.user_message
        )
        self.partial_prompt = PromptTemplate.from_template(partial_prompt)

    def get_prompt(self, prompt_example: str, table_desc: str, df_head: str):
        now = datetime.datetime.now(pytz.timezone('Asia/Singapore'))
        date_time = now.strftime('%d %B %Y, %H:%M')
        
        pandas_prompt = self.partial_prompt.partial(
            prompt_example=prompt_example,
            table_desc=table_desc,
            df_head=df_head,
            date_time=date_time
        )

        return pandas_prompt
