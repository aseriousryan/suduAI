from utils.llm import LargeLanguageModel

class BasePromptConstructor:
    def __init__(self, llm: LargeLanguageModel, system_message: str, user_message: str):
        self.prompt_template = llm.prompt_template
        self.system_message = system_message
        self.user_message = user_message

    def get_prompt(self):
        prompt = self.prompt_template.format(
            system_message=self.system_message,
            user_message=self.user_message
        )

        return prompt
