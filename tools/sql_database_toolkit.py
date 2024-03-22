from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from sqlalchemy import create_engine, text
from typing import List
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import Field
from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.tools import BaseTool
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from langchain_community.utilities.sql_database import SQLDatabase
import re

class DatabaseConnection:
    def __init__(self, db_user, db_password, db_host, db_name, db_port):
        connection_uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        self.engine = create_engine(connection_uri)

    def get_engine(self):
        print("Connection OK!")
        return self.engine
    
class SQLQueryInput(BaseModel):
    query: str

class SQLQueryTool(BaseTool):
    name = "SQLQueryTool"
    description = "A tool for executing SQL queries."
    
    # Conditional check for potential database modification
    def contains_forbidden_keywords(self, query):
            """
            Check if the query contains any SQL keywords that would modify the database.
            """
            forbidden_keywords = ['INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP', 'TRUNCATE', 'REPLACE', 'COMMENT', 'MERGE', 'GRANT', 'REVOKE', 'RENAME', 'LOCK TABLES', 'ROLLBACK']

            query_cleaned = re.sub(r"^\[SQL:\s*", "", query)
            query_cleaned = re.sub(r"\]$", "", query_cleaned)

            query_no_comments = re.sub(r"--.*?\n", "", query_cleaned, flags=re.MULTILINE)
            query_no_comments = re.sub(r"/\*.*?\*/", "", query_no_comments, flags=re.DOTALL).upper()

            return any(keyword in query_no_comments for keyword in forbidden_keywords)
   
    def _run(self, tool_input: SQLQueryInput, engine):
        """
        Executes a SQL query using the provided SQLAlchemy engine
        and returns the results.
        """
        query = tool_input.query.strip()

        # Validate that the query 
        if self.contains_forbidden_keywords(query):
            raise ValueError("The query contains forbidden keywords that could modify the database.")

        with engine.connect() as connection:
            result = connection.execute(text(query))
            column_names = list(result.keys())
            results = result.fetchall()
            return [{column_names[i]: value for i, value in enumerate(row)} for row in results]

    def run(self, tool_input: SQLQueryInput, engine):
        """Override the BaseTool.run to handle SQLQueryInput directly."""
        return self._run(tool_input=tool_input, engine=engine)


class TableSchemaTool(BaseTool):
    name = "TableSchemaTool"
    description = "A tool for retrieving table schema for a specific table from the database."

    def _run(self, engine):
        table_name = "delivery_order_listing"
        query = f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = '{table_name}'
            ORDER BY ordinal_position;
        """

        with engine.connect() as connection:
            result = connection.execute(text(query))
            schema = {}
            for row in result.fetchall():
                column_name, data_type = row
                schema[column_name] = data_type
            return schema

    def run(self, engine):
        return self._run(engine=engine)

class SQLDatabaseToolkit(BaseToolkit):
    db: SQLDatabase = Field(exclude=True)
    llm: BaseLanguageModel = Field(exclude=True)

    @property
    def dialect(self) -> str:
        return self.db.dialect

    class Config:
        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        list_sql_database_tool = ListSQLDatabaseTool(db=self.db)
        info_sql_database_tool_description = (
            "Input to this tool is a single table or comma-separated list of tables, output is the "
            "schema and sample rows for those tables. "
            "Example Input for list of tables: table1, table2, table3 "
            "Example Input for single table: table4"
        )
        info_sql_database_tool = InfoSQLDatabaseTool(
            db=self.db, description=info_sql_database_tool_description
        )
        query_sql_database_tool_description = (
            "Input to this tool is a detailed and correct MySQL query, output is a "
            "result from the database. If the query is not correct, an error message "
            "will be returned. If an error is returned, rewrite the query, check the "
            "query, and try again. If you encounter an issue with Unknown column "
            f"'xxxx' in 'field list' OR Unknown column 'xxxx' in 'where clause', use {info_sql_database_tool.name} "
            "to query the correct table column names from the given table schema."
        )
        query_sql_database_tool = QuerySQLDataBaseTool(
            db=self.db, description=query_sql_database_tool_description
        )
        query_sql_checker_tool_description = (
            "Use this tool to double check if your MySQL query is correct before executing "
            "it. Always use this tool before executing a query with "
            f"{query_sql_database_tool.name}!"
        )
        query_sql_checker_tool = QuerySQLCheckerTool(
            db=self.db, llm=self.llm, description=query_sql_checker_tool_description
        )

        tools = [
            # list_sql_database_tool,
            info_sql_database_tool,
            query_sql_database_tool,
            query_sql_checker_tool,
        ]
        
        return tools
    
    def get_context(self) -> dict:
        """Return db context that you may want in agent prompt."""
        return self.db.get_context()



if __name__ == "__main__":
    import sys
    import os

    # Add the parent directory of the current file to sys.path
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    sys.path.append(parent_dir)
    #testing
    from utils.llm import LargeLanguageModel
    from utils.common import LogData, read_yaml
    from langchain_community.utilities.sql_database import SQLDatabase
    from aserious_agent.sql_agent import SQLAgent
    import os

    db_connection = DatabaseConnection(db_user, db_password, db_host, db_name, db_port)

    engine = db_connection.get_engine()

    query_input = SQLQueryInput(query="SELECT * FROM sales_orders")

    query_tool = SQLQueryTool()

    results = query_tool.run(query_input, engine)

    print("Query results:")
    for row in results:
        print(row)


    schema_tool = TableSchemaTool()

    schema = schema_tool.run(engine)

    print("Table Schema:")
    for column_name, data_type in schema.items():
        print(f"{column_name}: {data_type}")

    import sys
    import os

    utils_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(utils_parent_dir)

    sql_db = SQLDatabase(engine)

    llm = LargeLanguageModel(**read_yaml(os.environ['model']))

    sql_toolkit = SQLDatabaseToolkit(db=sql_db, llm=llm.llm)

    # Test the get_tools method
    tools = sql_toolkit.get_tools()
    for tool in tools:
        print("Tools: ", tool.name)

    context = sql_toolkit.get_context()
    print("Context: ",context)

    table_info = context['table_info']
    table_info = table_info.replace('\\n', '\n').replace('\\t', '\t')

    print(table_info)
