from langchain.llms import LlamaCpp, OpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser



class LargeLanguageModel:
    def __init__(
        self, **kwargs
    ):
        if(kwargs['model_type'] == "llama-cpp"):
            callback_manager1 = CallbackManager([StreamingStdOutCallbackHandler()])
            self.llm = LlamaCpp(
                model_path=kwargs['model_path'],
                temperature=kwargs['temperature'],
                max_tokens=kwargs['max_tokens'],
                top_p=1,
                callback_manager=callback_manager1,
                verbose=True,
                streaming = True,
                n_gpu_layers=kwargs['n_gpu_layers'],
                n_ctx=kwargs['context_length']
            )
        else:

            self.llm = OpenAI(openai_api_key=kwargs['openai_api_key'])

        # simple runnable
        # self.prompt_template = PromptTemplate.from_template(kwargs['prompt_template'])
        # self.llm_runnable = self.prompt_template | self.llm | StrOutputParser()

        # self.instructions = kwargs['format_instructions']
        # self.prefix = kwargs['prefix']
        # self.suffix = kwargs['suffix']


        