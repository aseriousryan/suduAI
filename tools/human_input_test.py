# Import things that are needed generically
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool

class HumanQuery(BaseModel):
    query: str = Field(description="Should be a human query or human input")


@tool("human-query-response-tool", args_schema=HumanQuery, return_direct=True)
def HumanQueryResponse(query: str) -> str:
    """Find the ambiguity in the query or human input"""
    return "LangChain"


# keep this function only for testing out the agent 
# def create_human_input_tool(input_text):
#     tool = HumanInputTool()
#     args = HumanInputArgs(detailed_description="Please specify the timeframe and location for the delivery orders.")
#     tool_input = HumanInput(input=input_text, args=args)
#     result = tool.run(tool_input=tool_input)
#     return result

