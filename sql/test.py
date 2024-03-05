import os
import openai
from sqlalchemy import create_engine, text

# Importing api keys 
os.environ["OPENAI_API_KEY"] = "sk-yourkey"
openai.api_key = os.environ["OPENAI_API_KEY"]

# PostgreSQL
# db_user = "ryan.ho@aserious.co"
db_user = "postgres"
db_password = "6969ismylife"
db_host = "localhost"  
db_name = "DeCarton" 
db_port = "5432" 
connection_uri = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

engine = create_engine(connection_uri)

def test_connection(engine):
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT version();"))
            version = result.fetchone()
            print(f"Connected to PostgreSQL Server: {version[0]}")
        
            query_result = connection.execute(text("SELECT * FROM supplier_aging_report LIMIT 5;"))
            print("First 5 rows from the supplier_aging_report table:")
            for row in query_result:
                print(row)
                
            return True
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

# Call the test connection function
is_connected = test_connection(engine)
