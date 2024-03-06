# Import things that are needed generically
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from typing import ClassVar

from pydantic import BaseModel, Field

class HumanInputArgs(BaseModel):
    detailed_description: str = Field(
        default=None,
        description="A more detailed description of the question to clarify ambiguity.",
        example="I'm interested in the total number of delivery orders for last month."
    )
    example_questions: list[str] = Field(
        default=[],
        description="A list of example questions to guide the user in providing a more detailed query.",
        example=["What is the total number of delivery orders? ", "Can you specify the month for the delivery orders?"]
    )

class HumanInput(BaseModel):
    name: ClassVar[str] = 'Human Input Tool'
    description : ClassVar[str] = "Processes human input to clarify ambiguities."
    input: str
    args: HumanInputArgs = Field(
        default=None,
        description="Optional arguments to provide additional context or clarification requests."
    )

# keep this function only for testing out the agent 
def create_human_input_tool(input_text, detailed_description=None):

    args = HumanInputArgs(detailed_description=detailed_description) if detailed_description else None
    tool_input = HumanInput(input=input_text, args=args)
    tool = Output()
    return tool


class Output(BaseTool):
    # Metadata
    name = "human_input_tool"
    description = "A tool for conversational chatbots, to make follow-up questions if the user's question is ambiguous"

    args_schema = HumanInputArgs  # Assigning the args_schema

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, tool_input: HumanInput):
        """Asking user for follow-up question"""
        output = "Can you specify the question further?"
        if tool_input.args and tool_input.args.detailed_description:
            output += f" For example, {tool_input.args.detailed_description}"
        return output

    def run(self, tool_input: HumanInput):
        """Override the BaseTool.run to handle HumanInput directly."""
        return self._run(tool_input=tool_input)


if __name__ == "__main__":
    tool = Output()
    args = HumanInputArgs(detailed_description="Please specify the timeframe and location for the delivery orders.")
    tool_input = HumanInput(input="What is the total delivery order", args=args)
    result = tool.run(tool_input=tool_input)

    print(result)


