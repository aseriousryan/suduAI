from pymongo import MongoClient
import pandas as pd
import os

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

    def insert_many(self, data_dict, db_name=None, collection_name=None):
        if db_name: self.create_database(db_name)
        if collection_name: self.create_collection(collection_name)

        response = self.collection.insert_many(data_dict)

        return response.inserted_ids
        
    def insert_one(self, data, db_name=None, collection_name=None):
        if db_name: self.create_database(db_name)
        if collection_name: self.create_collection(collection_name)

        response = self.collection.insert_one(data)

        return response.inserted_id

    def find_all(self, db_name=None, collection_name=None, exclusion=None):
        if db_name: self.create_database(db_name)
        if collection_name: self.create_collection(collection_name)

        return pd.DataFrame(list(self.collection.find(projection=exclusion)))

    def find_one(self, db_name=None, collection_name=None, exclusion=None):
        if db_name: self.create_database(db_name)
        if collection_name: self.create_collection(collection_name)
        return pd.DataFrame(list(self.collection.find_one(collection_name, projection=exclusion)))
    
    def list_collections(self, db_name):
        if db_name: self.create_database(db_name)
        collection_list = self.db.list_collection_names()

        return collection_list
    
    def delete_many(self, query):
        response = self.collection.delete_many(query)
        return response.deleted_count
    
    def is_collection_exist(self, collection, db_name=None):
        if db_name: self.create_database(db_name)
        collection_list = self.db.list_collection_names()

        return collection in collection_list
    
    def get_table_desc(self, database_name, collection):
        # database name is the database name of where the table is stored, not the desc table database
        # collection is the name of the collection that we want the description
        df_desc = self.find_all(os.environ['mongodb_table_descriptor'], database_name)
        df_desc = df_desc.loc[df_desc['collection'] == collection]
        if len(df_desc) == 0:
            table_desc = ''
        else:
            table_desc = df_desc['description'].iloc[0]
            table_desc = f'The following is information about the table df:\n{table_desc}\n'

        return table_desc
