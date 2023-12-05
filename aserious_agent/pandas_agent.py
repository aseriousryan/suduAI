# """Agent for working with pandas objects."""
# from typing import Any, Dict, List, Optional, Sequence, Tuple

# from langchain.agents.agent import AgentExecutor, BaseSingleActionAgent
# # from langchain.agents.mrkl.base import ZeroShotAgent
# from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
# from langchain.agents.types import AgentType
# from langchain.callbacks.base import BaseCallbackManager
# from langchain.chains.llm import LLMChain
# from langchain.schema import BasePromptTemplate, HumanMessage
# from langchain.schema.language_model import BaseLanguageModel
# from langchain.schema.messages import SystemMessage
# from langchain.tools import BaseTool
# from langchain.prompts import PromptTemplate
# from langchain.agents import Tool


# from langchain_experimental.agents.agent_toolkits.pandas.prompt import (
#     FUNCTIONS_WITH_DF,
#     FUNCTIONS_WITH_MULTI_DF,
#     MULTI_DF_PREFIX,
#     MULTI_DF_PREFIX_FUNCTIONS,
#     PREFIX,
#     PREFIX_FUNCTIONS,
#     SUFFIX_NO_DF,
#     SUFFIX_WITH_DF,
#     SUFFIX_WITH_MULTI_DF,
# )
# from langchain_experimental.tools.python.tool import PythonAstREPLTool
# from aserious_agent.base import ZeroShotAgent
# from langchain.prompts import BaseChatPromptTemplate


# # PREFIX = """### System:
# # You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
# # You must set the Action as python_repl_ast tool to answer the question posed to you:"""

# # MULTI_DF_PREFIX = """You are working with {num_dfs} pandas dataframes in Python named df1, df2, etc. You 
# # should use the tools below to answer the question posed of you:"""

# # SUFFIX_NO_DF = """
# # Begin!
# # Question: {input}
# # {agent_scratchpad}"""

# # SUFFIX_WITH_DF = """
# # This is the result of `print(df.head())`:
# # {df_head}

# # Begin!

# # ### User:
# # Question: {input}
# # {agent_scratchpad} 

# # ### Assistant:"""

# # SUFFIX_WITH_MULTI_DF = """
# # This is the result of `print(df.head())` for each dataframe:
# # {dfs_head}

# # Begin!
# # Question: {input}
# # {agent_scratchpad} [/INST]"""

# # PREFIX_FUNCTIONS = """### System:
# # You are working with a pandas dataframe in Python. The name of the dataframe is `df`."""

# # MULTI_DF_PREFIX_FUNCTIONS = """
# # You are working with {num_dfs} pandas dataframes in Python named df1, df2, etc."""

# # FUNCTIONS_WITH_DF = """
# # This is the result of `print(df.head())`:
# # {df_head}"""

# # FUNCTIONS_WITH_MULTI_DF = """
# # This is the result of `print(df.head())` for each dataframe:
# # {dfs_head}"""

# prefix = """### System:
# You are working with a pandas dataframe in Python. The name of the dataframe is 'df'.
# You should use the tools below to answer the question posed of you:
# """

# suffix= """Begin! 
# ### User:
# Question: {input}
# {agent_scratchpad} 

# ### Assistant:
# """

# input_variables=["input", "agent_scratchpad"],

# # Set up a prompt template
# class CustomPromptTemplate(BaseChatPromptTemplate):
#     # The template to use
#     template: str
#     # The list of tools available
#     tools: List[Tool]

#     def format_messages(self, **kwargs) -> str:
#         # Get the intermediate steps (AgentAction, Observation tuples)
#         # Format them in a particular way
#         intermediate_steps = kwargs.pop("intermediate_steps")
#         thoughts = ""
#         for action, observation in intermediate_steps:
#             thoughts += action.log
#             thoughts += f"\nObservation: {observation}\nThought: "
#         # Set the agent_scratchpad variable to that value
#         kwargs["agent_scratchpad"] = thoughts
#         # Create a tools variable from the list of tools provided
#         kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
#         # Create a list of tool names for the tools provided
#         kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
#         formatted = self.template.format(**kwargs)
#         return [HumanMessage(content=formatted)]

# # Set up the base template
# template = """Complete the objective as best you can. You have access to the following tools:

# {tools}

# Use the following format:

# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question

# These were previous tasks you completed:



# Begin!

# Question: {input}
# {agent_scratchpad}"""


# def _get_multi_prompt(
#     dfs: List[Any],
#     prefix: Optional[str] = None,
#     suffix: Optional[str] = None,
#     input_variables: Optional[List[str]] = None,
#     include_df_in_prompt: Optional[bool] = True,
#     number_of_head_rows: int = 5,
# ) -> Tuple[BasePromptTemplate, List[PythonAstREPLTool]]:
#     num_dfs = len(dfs)
#     if suffix is not None:
#         suffix_to_use = suffix
#         include_dfs_head = True
#     elif include_df_in_prompt:
#         suffix_to_use = SUFFIX_WITH_MULTI_DF
#         include_dfs_head = True
#     else:
#         suffix_to_use = SUFFIX_NO_DF
#         include_dfs_head = False
#     if input_variables is None:
#         input_variables = ["input", "agent_scratchpad", "num_dfs"]
#         if include_dfs_head:
#             input_variables += ["dfs_head"]

#     if prefix is None:
#         prefix = MULTI_DF_PREFIX

#     df_locals = {}
#     for i, dataframe in enumerate(dfs):
#         df_locals[f"df{i + 1}"] = dataframe
#     tools = [PythonAstREPLTool(locals=df_locals)]

#     prompt = ZeroShotAgent.create_prompt(
#         tools, prefix=prefix, suffix=suffix_to_use, input_variables=input_variables
#     )

#     partial_prompt = prompt.partial()
#     if "dfs_head" in input_variables:
#         dfs_head = "\n\n".join([d.head(number_of_head_rows).to_markdown() for d in dfs])
#         partial_prompt = partial_prompt.partial(num_dfs=str(num_dfs), dfs_head=dfs_head)
#     if "num_dfs" in input_variables:
#         partial_prompt = partial_prompt.partial(num_dfs=str(num_dfs))
#     return partial_prompt, tools


# def _get_single_prompt(
#     df: Any,
#     prefix: Optional[str] = None,
#     suffix: Optional[str] = None,
#     input_variables: Optional[List[str]] = None,
#     include_df_in_prompt: Optional[bool] = True,
#     number_of_head_rows: int = 5,
# ) -> Tuple[BasePromptTemplate, List[PythonAstREPLTool]]:
#     if suffix is not None:
#         suffix_to_use = suffix
#         include_df_head = True
#     elif include_df_in_prompt:
#         suffix_to_use = SUFFIX_WITH_DF
#         include_df_head = True
#     else:
#         suffix_to_use = SUFFIX_NO_DF
#         include_df_head = False

#     if input_variables is None:
#         input_variables = ["input", "agent_scratchpad"]
#         if include_df_head:
#             input_variables += ["df_head"]

#     if prefix is None:
#         prefix = PREFIX

#     tools = [PythonAstREPLTool(locals={"df": df})]

#     prompt = ZeroShotAgent.create_prompt(
#         tools, prefix=prefix, suffix=suffix_to_use, input_variables=input_variables
#     )

#     partial_prompt = prompt.partial()
#     if "df_head" in input_variables:
#         partial_prompt = partial_prompt.partial(
#             df_head=str(df.head(number_of_head_rows).to_markdown())
#         )
#     return partial_prompt, tools


# def _get_prompt_and_tools(
#     df: Any,
#     prefix: Optional[str] = None,
#     suffix: Optional[str] = None,
#     input_variables: Optional[List[str]] = None,
#     include_df_in_prompt: Optional[bool] = True,
#     number_of_head_rows: int = 5,
# ) -> Tuple[CustomPromptTemplate, List[PythonAstREPLTool]]:
#     try:
#         import pandas as pd

#         pd.set_option("display.max_columns", None)
#     except ImportError:
#         raise ImportError(
#             "pandas package not found, please install with `pip install pandas`"
#         )

#     if include_df_in_prompt is not None and suffix is not None:
#         raise ValueError("If suffix is specified, include_df_in_prompt should not be.")

#     if isinstance(df, list):
#         for item in df:
#             if not isinstance(item, pd.DataFrame):
#                 raise ValueError(f"Expected pandas object, got {type(df)}")
#         return _get_multi_prompt(
#             df,
#             prefix=prefix,
#             suffix=suffix,
#             input_variables=input_variables,
#             include_df_in_prompt=include_df_in_prompt,
#             number_of_head_rows=number_of_head_rows,
#         )
#     else:
#         if not isinstance(df, pd.DataFrame):
#             raise ValueError(f"Expected pandas object, got {type(df)}")
#         return _get_single_prompt(
#             df,
#             prefix=prefix,
#             suffix=suffix,
#             input_variables=input_variables,
#             include_df_in_prompt=include_df_in_prompt,
#             number_of_head_rows=number_of_head_rows,
#         )


# def _get_functions_single_prompt(
#     df: Any,
#     prefix: Optional[str] = None,
#     suffix: Optional[str] = None,
#     include_df_in_prompt: Optional[bool] = True,
#     number_of_head_rows: int = 5,
# ) -> Tuple[BasePromptTemplate, List[PythonAstREPLTool]]:
#     if suffix is not None:
#         suffix_to_use = suffix
#         if include_df_in_prompt:
#             suffix_to_use = suffix_to_use.format(
#                 df_head=str(df.head(number_of_head_rows).to_markdown())
#             )
#     elif include_df_in_prompt:
#         suffix_to_use = FUNCTIONS_WITH_DF.format(
#             df_head=str(df.head(number_of_head_rows).to_markdown())
#         )
#     else:
#         suffix_to_use = ""

#     if prefix is None:
#         prefix = PREFIX_FUNCTIONS

#     tools = [PythonAstREPLTool(locals={"df": df})]
#     system_message = SystemMessage(content=prefix + suffix_to_use)
#     prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
#     return prompt, tools


# def _get_functions_multi_prompt(
#     dfs: Any,
#     prefix: Optional[str] = None,
#     suffix: Optional[str] = None,
#     include_df_in_prompt: Optional[bool] = True,
#     number_of_head_rows: int = 5,
# ) -> Tuple[BasePromptTemplate, List[PythonAstREPLTool]]:
#     if suffix is not None:
#         suffix_to_use = suffix
#         if include_df_in_prompt:
#             dfs_head = "\n\n".join(
#                 [d.head(number_of_head_rows).to_markdown() for d in dfs]
#             )
#             suffix_to_use = suffix_to_use.format(
#                 dfs_head=dfs_head,
#             )
#     elif include_df_in_prompt:
#         dfs_head = "\n\n".join([d.head(number_of_head_rows).to_markdown() for d in dfs])
#         suffix_to_use = FUNCTIONS_WITH_MULTI_DF.format(
#             dfs_head=dfs_head,
#         )
#     else:
#         suffix_to_use = ""

#     if prefix is None:
#         prefix = MULTI_DF_PREFIX_FUNCTIONS
#     prefix = prefix.format(num_dfs=str(len(dfs)))

#     df_locals = {}
#     for i, dataframe in enumerate(dfs):
#         df_locals[f"df{i + 1}"] = dataframe
#     tools = [PythonAstREPLTool(locals=df_locals)]
#     system_message = SystemMessage(content=prefix + suffix_to_use)
#     prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
#     return prompt, tools


# def _get_functions_prompt_and_tools(
#     df: Any,
#     prefix: Optional[str] = None,
#     suffix: Optional[str] = None,
#     input_variables: Optional[List[str]] = None,
#     include_df_in_prompt: Optional[bool] = True,
#     number_of_head_rows: int = 5,
# ) -> Tuple[BasePromptTemplate, List[PythonAstREPLTool]]:
#     try:
#         import pandas as pd

#         pd.set_option("display.max_columns", None)
#     except ImportError:
#         raise ImportError(
#             "pandas package not found, please install with `pip install pandas`"
#         )
#     if input_variables is not None:
#         raise ValueError("`input_variables` is not supported at the moment.")

#     if include_df_in_prompt is not None and suffix is not None:
#         raise ValueError("If suffix is specified, include_df_in_prompt should not be.")

#     if isinstance(df, list):
#         for item in df:
#             if not isinstance(item, pd.DataFrame):
#                 raise ValueError(f"Expected pandas object, got {type(df)}")
#         return _get_functions_multi_prompt(
#             df,
#             prefix=prefix,
#             suffix=suffix,
#             include_df_in_prompt=include_df_in_prompt,
#             number_of_head_rows=number_of_head_rows,
#         )
#     else:
#         if not isinstance(df, pd.DataFrame):
#             raise ValueError(f"Expected pandas object, got {type(df)}")
#         return _get_functions_single_prompt(
#             df,
#             prefix=prefix,
#             suffix=suffix,
#             include_df_in_prompt=include_df_in_prompt,
#             number_of_head_rows=number_of_head_rows,
#         )




# def create_pandas_dataframe_agent(
#     llm: BaseLanguageModel,
#     df: Any,
#     agent_type: AgentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     callback_manager: Optional[BaseCallbackManager] = None,
#     prefix: Optional[str] = None,
#     suffix: Optional[str] = None,
#     input_variables: Optional[List[str]] = None,
#     verbose: bool = False,
#     return_intermediate_steps: bool = False,
#     max_iterations: Optional[int] = 15,
#     max_execution_time: Optional[float] = None,
#     early_stopping_method: str = "force",
#     agent_executor_kwargs: Optional[Dict[str, Any]] = None,
#     include_df_in_prompt: Optional[bool] = True,
#     number_of_head_rows: int = 5,
#     extra_tools: Sequence[BaseTool] = (),
#     **kwargs: Dict[str, Any],
# ) -> AgentExecutor:
#     """Construct a pandas agent from an LLM and dataframe."""
#     agent: BaseSingleActionAgent
#     if agent_type == AgentType.ZERO_SHOT_REACT_DESCRIPTION:
#         prompt, base_tools = _get_prompt_and_tools(
#             df,
#             prefix=prefix,
#             suffix=suffix,
#             input_variables=input_variables,
#             include_df_in_prompt=include_df_in_prompt,
#             number_of_head_rows=number_of_head_rows,
#         )

        

#         tools = base_tools + list(extra_tools)
#         llm_chain = LLMChain(
#             llm=llm,
#             prompt=prompt,
#             # callback_manager=callback_manager,
#         )
#         tool_names = [tool.name for tool in tools]
#         agent = ZeroShotAgent(
#             llm_chain=llm_chain,
#             allowed_tools=tool_names,
#             callback_manager=callback_manager,
#             **kwargs,
#         )
#     elif agent_type == AgentType.OPENAI_FUNCTIONS:
#         _prompt, base_tools = _get_functions_prompt_and_tools(
#             df,
#             prefix=prefix,
#             suffix=suffix,
#             input_variables=input_variables,
#             include_df_in_prompt=include_df_in_prompt,
#             number_of_head_rows=number_of_head_rows,
#         )
#         tools = base_tools + list(extra_tools)
#         agent = OpenAIFunctionsAgent(
#             llm=llm,
#             prompt=_prompt,
#             tools=tools,
#             callback_manager=callback_manager,
#             **kwargs,
#         )
#     else:
#         raise ValueError(f"Agent type {agent_type} not supported at the moment.")
#     return AgentExecutor.from_agent_and_tools(
#         agent=agent,
#         tools=tools,
#         callback_manager=callback_manager,
#         verbose=verbose,
#         return_intermediate_steps=return_intermediate_steps,
#         max_iterations=max_iterations,
#         max_execution_time=max_execution_time,
#         early_stopping_method=early_stopping_method,
#         handle_parsing_errors=True,
#         **(agent_executor_kwargs or {}),
#     )


"""Agent for working with pandas objects."""
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain.agents.agent import AgentExecutor, BaseSingleActionAgent
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.agents.types import AgentType
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import SystemMessage
from langchain.tools import BaseTool

from langchain_experimental.agents.agent_toolkits.pandas.prompt import (
    FUNCTIONS_WITH_DF,
    FUNCTIONS_WITH_MULTI_DF,
    MULTI_DF_PREFIX,
    MULTI_DF_PREFIX_FUNCTIONS,
    PREFIX,
    PREFIX_FUNCTIONS,
    SUFFIX_NO_DF,
    SUFFIX_WITH_DF,
    SUFFIX_WITH_MULTI_DF,
)
from langchain_experimental.tools.python.tool import PythonAstREPLTool



def _get_multi_prompt(
    dfs: List[Any],
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    include_df_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 5,
) -> Tuple[BasePromptTemplate, List[PythonAstREPLTool]]:
    num_dfs = len(dfs)
    if suffix is not None:
        suffix_to_use = suffix
        include_dfs_head = True
    elif include_df_in_prompt:
        suffix_to_use = SUFFIX_WITH_MULTI_DF
        include_dfs_head = True
    else:
        suffix_to_use = SUFFIX_NO_DF
        include_dfs_head = False
    if input_variables is None:
        input_variables = ["input", "agent_scratchpad", "num_dfs"]
        if include_dfs_head:
            input_variables += ["dfs_head"]

    if prefix is None:
        prefix = MULTI_DF_PREFIX

    df_locals = {}
    for i, dataframe in enumerate(dfs):
        df_locals[f"df{i + 1}"] = dataframe
    tools = [PythonAstREPLTool(locals=df_locals)]

    prompt = ZeroShotAgent.create_prompt(
        tools, prefix=prefix, suffix=suffix_to_use, input_variables=input_variables
    )

    partial_prompt = prompt.partial()
    if "dfs_head" in input_variables:
        dfs_head = "\n\n".join([d.head(number_of_head_rows).to_markdown() for d in dfs])
        partial_prompt = partial_prompt.partial(num_dfs=str(num_dfs), dfs_head=dfs_head)
    if "num_dfs" in input_variables:
        partial_prompt = partial_prompt.partial(num_dfs=str(num_dfs))
    return partial_prompt, tools


def _get_single_prompt(
    df: Any,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    include_df_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 5,
) -> Tuple[BasePromptTemplate, List[PythonAstREPLTool]]:
    if suffix is not None:
        suffix_to_use = suffix
        include_df_head = True
    elif include_df_in_prompt:
        suffix_to_use = SUFFIX_WITH_DF
        include_df_head = True
    else:
        suffix_to_use = SUFFIX_NO_DF
        include_df_head = False

    if input_variables is None:
        input_variables = ["input", "agent_scratchpad"]
        if include_df_head:
            input_variables += ["df_head"]

    if prefix is None:
        prefix = PREFIX

    tools = [PythonAstREPLTool(locals={"df": df})]

    prompt = ZeroShotAgent.create_prompt(
        tools, prefix=prefix, suffix=suffix_to_use, input_variables=input_variables
    )

    partial_prompt = prompt.partial()
    if "df_head" in input_variables:
        partial_prompt = partial_prompt.partial(
            df_head=str(df.head(number_of_head_rows).to_markdown())
        )
    return partial_prompt, tools


def _get_prompt_and_tools(
    df: Any,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    include_df_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 5,
) -> Tuple[BasePromptTemplate, List[PythonAstREPLTool]]:
    try:
        import pandas as pd

        pd.set_option("display.max_columns", None)
    except ImportError:
        raise ImportError(
            "pandas package not found, please install with `pip install pandas`"
        )

    if include_df_in_prompt is not None and suffix is not None:
        raise ValueError("If suffix is specified, include_df_in_prompt should not be.")

    if isinstance(df, list):
        for item in df:
            if not isinstance(item, pd.DataFrame):
                raise ValueError(f"Expected pandas object, got {type(df)}")
        return _get_multi_prompt(
            df,
            prefix=prefix,
            suffix=suffix,
            input_variables=input_variables,
            include_df_in_prompt=include_df_in_prompt,
            number_of_head_rows=number_of_head_rows,
        )
    else:
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected pandas object, got {type(df)}")
        return _get_single_prompt(
            df,
            prefix=prefix,
            suffix=suffix,
            input_variables=input_variables,
            include_df_in_prompt=include_df_in_prompt,
            number_of_head_rows=number_of_head_rows,
        )


def _get_functions_single_prompt(
    df: Any,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    include_df_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 5,
) -> Tuple[BasePromptTemplate, List[PythonAstREPLTool]]:
    if suffix is not None:
        suffix_to_use = suffix
        if include_df_in_prompt:
            suffix_to_use = suffix_to_use.format(
                df_head=str(df.head(number_of_head_rows).to_markdown())
            )
    elif include_df_in_prompt:
        suffix_to_use = FUNCTIONS_WITH_DF.format(
            df_head=str(df.head(number_of_head_rows).to_markdown())
        )
    else:
        suffix_to_use = ""

    if prefix is None:
        prefix = PREFIX_FUNCTIONS

    tools = [PythonAstREPLTool(locals={"df": df})]
    system_message = SystemMessage(content=prefix + suffix_to_use)
    prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
    return prompt, tools


def _get_functions_multi_prompt(
    dfs: Any,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    include_df_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 5,
) -> Tuple[BasePromptTemplate, List[PythonAstREPLTool]]:
    if suffix is not None:
        suffix_to_use = suffix
        if include_df_in_prompt:
            dfs_head = "\n\n".join(
                [d.head(number_of_head_rows).to_markdown() for d in dfs]
            )
            suffix_to_use = suffix_to_use.format(
                dfs_head=dfs_head,
            )
    elif include_df_in_prompt:
        dfs_head = "\n\n".join([d.head(number_of_head_rows).to_markdown() for d in dfs])
        suffix_to_use = FUNCTIONS_WITH_MULTI_DF.format(
            dfs_head=dfs_head,
        )
    else:
        suffix_to_use = ""

    if prefix is None:
        prefix = MULTI_DF_PREFIX_FUNCTIONS
    prefix = prefix.format(num_dfs=str(len(dfs)))

    df_locals = {}
    for i, dataframe in enumerate(dfs):
        df_locals[f"df{i + 1}"] = dataframe
    tools = [PythonAstREPLTool(locals=df_locals)]
    system_message = SystemMessage(content=prefix + suffix_to_use)
    prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
    return prompt, tools


def _get_functions_prompt_and_tools(
    df: Any,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    include_df_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 5,
) -> Tuple[BasePromptTemplate, List[PythonAstREPLTool]]:
    try:
        import pandas as pd

        pd.set_option("display.max_columns", None)
    except ImportError:
        raise ImportError(
            "pandas package not found, please install with `pip install pandas`"
        )
    if input_variables is not None:
        raise ValueError("`input_variables` is not supported at the moment.")

    if include_df_in_prompt is not None and suffix is not None:
        raise ValueError("If suffix is specified, include_df_in_prompt should not be.")

    if isinstance(df, list):
        for item in df:
            if not isinstance(item, pd.DataFrame):
                raise ValueError(f"Expected pandas object, got {type(df)}")
        return _get_functions_multi_prompt(
            df,
            prefix=prefix,
            suffix=suffix,
            include_df_in_prompt=include_df_in_prompt,
            number_of_head_rows=number_of_head_rows,
        )
    else:
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected pandas object, got {type(df)}")
        return _get_functions_single_prompt(
            df,
            prefix=prefix,
            suffix=suffix,
            include_df_in_prompt=include_df_in_prompt,
            number_of_head_rows=number_of_head_rows,
        )


def create_pandas_dataframe_agent(
    llm: BaseLanguageModel,
    df: Any,
    agent_type: AgentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    verbose: bool = False,
    return_intermediate_steps: bool = False,
    max_iterations: Optional[int] = 15,
    max_execution_time: Optional[float] = None,
    early_stopping_method: str = "force",
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    include_df_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 5,
    extra_tools: Sequence[BaseTool] = (),
    **kwargs: Dict[str, Any],
) -> AgentExecutor:
    """Construct a pandas agent from an LLM and dataframe."""
    agent: BaseSingleActionAgent
    if agent_type == AgentType.ZERO_SHOT_REACT_DESCRIPTION:
        prompt, base_tools = _get_prompt_and_tools(
            df,
            prefix=prefix,
            suffix=suffix,
            input_variables=input_variables,
            include_df_in_prompt=include_df_in_prompt,
            number_of_head_rows=number_of_head_rows,
        )
        tools = base_tools + list(extra_tools)
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        agent = ZeroShotAgent(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            callback_manager=callback_manager,
            **kwargs,
        )
    elif agent_type == AgentType.OPENAI_FUNCTIONS:
        _prompt, base_tools = _get_functions_prompt_and_tools(
            df,
            prefix=prefix,
            suffix=suffix,
            input_variables=input_variables,
            include_df_in_prompt=include_df_in_prompt,
            number_of_head_rows=number_of_head_rows,
        )
        tools = base_tools + list(extra_tools)
        agent = OpenAIFunctionsAgent(
            llm=llm,
            prompt=_prompt,
            tools=tools,
            callback_manager=callback_manager,
            **kwargs,
        )
    else:
        raise ValueError(f"Agent type {agent_type} not supported at the moment.")
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        verbose=verbose,
        return_intermediate_steps=return_intermediate_steps,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        early_stopping_method=early_stopping_method,
        **(agent_executor_kwargs or {}),
    )