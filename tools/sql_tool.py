from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, tool
from sqlalchemy import create_engine, text
from typing import Any, Dict, List, Literal, Optional, Sequence, Union
from sqlalchemy.engine import Engine

class DatabaseConnection:
    def __init__(self, db_user, db_password, db_host, db_name, db_port):
        connection_uri = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        self.engine = create_engine(connection_uri)

    def get_engine(self):
        return self.engine

class SQLQueryInput(BaseModel):
    sql_query: str = Field(description="code snippet to run")

class SQLQueryTool(BaseTool): 
    name: str = "sql_db_query"
    description: str = """
    A SQL shell. Use this to execute SQL query
    Input should be a valid SQL query with double quotes on all column names and alias names if any. 
    Do not use triple backticks in SQL shell.
    When using this tool, sometimes output is abbreviated - make sure it does not look abbreviated before using it in your answer.
    """

    engine: Any  # Add this line to declare the engine attribute

    def __init__(self, engine: Any, **kwargs: Any):
        super().__init__(**kwargs)
        self.engine = engine

    def _run(self, tool_input: Union[SQLQueryInput, str], **kwargs: Any) -> List[Dict[str, Any]]:
        if isinstance(tool_input, SQLQueryInput):
            query = tool_input.sql_query
        elif isinstance(tool_input, str):
            query = tool_input
        else:
            raise ValueError("Unsupported input type. Use SQLQueryInput or str.")

        with self.engine.connect() as connection:
            result = connection.execute(text(query))
            column_names = list(result.keys())
            results = result.fetchall()
            return [{column_names[i]: value for i, value in enumerate(row)} for row in results]

    def run(self, tool_input: Union[SQLQueryInput, str], **kwargs: Any) -> List[Dict[str, Any]]:
        return self._run(tool_input=tool_input, **kwargs)


if __name__ == "__main__":
    # Example usage
    db_connection = DatabaseConnection(db_user="postgres", db_password="6969ismylife", db_host="localhost", db_name="DeCarton", db_port="5432")

    # Create an SQLQueryTool instance
    sql_tool = SQLQueryTool(engine=db_connection.get_engine())

    # Test the SQLQueryTool with an example SQL query
    example_query = "SELECT EXTRACT(YEAR FROM \"Doc. Date\") AS \"Year\", EXTRACT(MONTH FROM \"Doc. Date\") AS \"Month\", SUM(\"Amount\") AS \"Total Sales\" FROM invoice_invoice_collection WHERE EXTRACT(YEAR FROM \"Doc. Date\") IN (2023, 2024) GROUP BY EXTRACT(YEAR FROM \"Doc. Date\"), EXTRACT(MONTH FROM \"Doc. Date\") ORDER BY SUM(\"Amount\") DESC LIMIT 1;"

    input_data = SQLQueryInput(sql_query=example_query)

    # Run the SQLQueryTool and print the result
    result = sql_tool.run(input_data)
    print(result)