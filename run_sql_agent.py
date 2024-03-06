# Import necessary modules
from utils.llm import LargeLanguageModel
from aserious_agent.sql_agent import SQLAgent
from tools.sql_database_toolkit import SQLQueryTool, SQLDatabaseToolkit, DatabaseConnection
from langchain.sql_database import SQLDatabase
from dotenv import load_dotenv
from utils.common import (
    tokenize, ENV, read_yaml, parse_langchain_debug_log, LogData
)
import os
from utils.redirect_print import RedirectPrint


load_dotenv(f'./.env.{ENV}')

# Define the directory and file path
directory = "debug_log"
file_path = os.path.join(directory, "log.txt")

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

llm = LargeLanguageModel(**read_yaml(os.environ['model']))

data_logger = LogData()

db_user = os.getenv('db_user')
db_password = os.getenv('db_password')
db_host = os.getenv('db_host')
db_name = os.getenv('db_name')
db_port = int(os.getenv('db_port'))

db_connection = DatabaseConnection(db_user, db_password, db_host, db_name, db_port)

engine = db_connection.get_engine()

sql_db = SQLDatabase(engine)

sql_agent = SQLAgent(llm=llm, db=sql_db, data_logger=data_logger)

user_query = "Count the amounts of total delivery order in January?"
rp = RedirectPrint()
rp.start()
result = sql_agent.run_agent(user_query=user_query)
debug_log = parse_langchain_debug_log(rp.get_output())
#Write the output to the file
with open(file_path, "w") as file:
    file.write(debug_log)
rp.stop()

print(result)
