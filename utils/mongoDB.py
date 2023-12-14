from pymongo import MongoClient
import pandas as pd

class MongoDBController:
    def __init__(self, host, port, username, password, db_name=None):
        self.client = MongoClient(host, port=port, username=username, password=password, connect=False)
        if db_name:
            self.db = self.client[db_name]
        else:
            self.db = None

        self.collection = None
        
    def create_database(self, db_name):
        self.db = self.client[db_name]

        return self.db

    def create_collection(self, collection_name):
        # create collection if it does not exist, else retrieve
        self.collection = self.db[collection_name]

        return self.collection

    def insert_many(self, data_dict):
        response = self.collection.insert_many(data_dict)

        return response.inserted_ids
        
    def insert_one(self, data):
        response = self.collection.insert_one(data)

        return response.inserted_id
        
    def find_all(self):
        return pd.DataFrame(list(self.collection.find()))

    def delete_many(self, query):
        response = self.collection.delete_many(query)
        return response.deleted_count
