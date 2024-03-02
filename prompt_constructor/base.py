from utils.llm import LargeLanguageModel

class BasePromptConstructor:
    def __init__(self, llm: LargeLanguageModel, system_prompt: str):
        self.prompt_template = llm.prompt_template
        self.system_prompt = system_prompt

    def get_prompt(self):
        prompt = self.prompt_template.format(
            system_message=self.system_prompt
        )

        return prompt
