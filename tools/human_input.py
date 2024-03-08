# Import things that are needed generically
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from typing import ClassVar

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

class HumanInputTool(BaseTool):
    name = "Human_Input_Tool"
    description = "A tool for conversational chatbots to make follow-up questions if the user's question is ambiguous, for example, if the question does not provide specific information in the question"

    args_schema = HumanInputArgs
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, tool_input: HumanInput, **kwargs):
        """Asking user for a follow-up question with optional detailed description."""
        output = "Can you specify the question further?"
        if tool_input.args and tool_input.args.detailed_description:
            output += f" For example, {tool_input.args.detailed_description}"
        return output

    def run(self, tool_input: HumanInput, **kwargs):
        """Override the BaseTool.run to handle HumanInput directly, ignoring unexpected keyword arguments."""
        # return self._run(tool_input=tool_input)
        return "July"

# print(create_human_input_tool("what is the total delivery orders", "Please specify the timeframe and location for the delivery orders."))
if __name__ == "__main__":
    tool = HumanInputTool()
    args = HumanInputArgs(detailed_description="Please specify the timeframe and location for the delivery orders.")
    tool_input = HumanInput(input="What is the total delivery order", args=args)
    result = tool.run(tool_input=tool_input)

    print(result)


