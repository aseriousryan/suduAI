# MongoDBOperations.py
from pymongo import MongoClient
import pandas as pd

class MongoDBOperations:
    def __init__(self, host, port, username, password):
        try:
            self.client = MongoClient(host, port= port, username = username,password = password, connect=False)
        except Exception as e:
            print(f"Error connecting to MongoDB server: {e}")

    def create_database(self, db_name):
        # Creates a new database
        if db_name:
            return self.client[db_name]
        else:
            raise ValueError("Database name cannot be None")

    def create_collection(self, db_name, collection_name):
        # Creates a new collection in the specified database
        if db_name and collection_name:
            db = self.create_database(db_name)
            return db[collection_name]
        else:
            raise ValueError("Database name and collection name cannot be None")

    def insert_many(self, db_name, collection_name, data_dict):
        # Inserts multiple documents into the specified collection
        if db_name and collection_name and data_dict:
            collection = self.create_collection(db_name, collection_name)
            return collection.insert_many(data_dict)
        else:
            raise ValueError("Database name, collection name, and data dictionary cannot be None")

    def insert_one(self, data):
        if data:
            evaluator_db = self.client['logs']
            evaluator_collection = evaluator_db['de-carton']
            return evaluator_collection.insert_one(data)
        else:
            raise ValueError("Output data cannot be None")
        
    def find_all(self, db_name):
        #Finds all documents in the specified collection
        if db_name and collection_name:
            db = self.client[db_name]

            dataframes = []
            for collection_name in db.list_collection_names():
                collection = db[collection_name]
                df = pd.DataFrame(list(collection.find()))
                dataframes.append(df)

            return dataframes
        else:
            raise ValueError("Database name cannot be None")

    def delete_many(self, db_name, collection_name, query):
        # Deletes multiple documents from the specified collection
        if db_name and collection_name and query:
            db = self.create_database(db_name)
            collection = db[collection_name]
            return collection.delete_many(query)
        else:
            raise ValueError("Database name, collection name, and query cannot be None")

