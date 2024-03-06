from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from sqlalchemy import create_engine, text


# PostgreSQL connection stream
# db_user = "ryan.ho@aserious.co"
db_user = "postgres"
db_password = "6969ismylife"
db_host = "localhost"  
db_name = "DeCarton" 
db_port = "5432" 
connection_uri = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(connection_uri)

# Pydantic base tool in ensuring formatting
class SQLQueryInput(BaseModel):
    query: str

class SQLQueryTool(BaseTool):
    name = "SQLQueryTool"
    description = "A tool for executing SQL queries."
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _run(self, tool_input: SQLQueryInput):
        """Executes a SQL query using a passed SQLAlchemy engine and returns the results."""
        query = tool_input.query  
        
        with engine.connect() as connection:
            # Execution of query
            result = connection.execute(text(query))
            column_names = list(result.keys())
            results = result.fetchall()
            return [{column_names[i]: value for i, value in enumerate(row)} for row in results]


    def run(self, tool_input: SQLQueryInput):
        """Override the BaseTool.run to handle SQLQueryInput directly."""
        return self._run(tool_input=tool_input)


if __name__ == "__main__":
    tool = SQLQueryTool()
    tool_input = SQLQueryInput(query="SELECT * FROM customer_aging_report LIMIT 5;")
    results = tool.run(tool_input=tool_input)  
    print("Query Results:")
    for row in results:
        print(row)

