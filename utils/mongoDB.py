from pymongo import MongoClient
import pandas as pd
import os
from bson.json_util import dumps
from preprocessors import row_descriptor

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
    
    def get_table_desc_collection(self, database_name, collection):
        # database name is the database name of where the table is stored, not the desc table database
        # collection is the name of the collection that we want the description
        df_desc = self.find_all(os.environ['mongodb_table_descriptor'], database_name)
        df_collection_name = df_desc.loc[df_desc['collection'] == collection]
        return df_collection_name

    def get_collection_data_types(self, db_name, collection_name):
        existing_data = self.find_all(db_name, collection_name)
        existing_data = existing_data.drop(columns=['_id'], errors='ignore')
        data_types = existing_data.dtypes.to_dict()
        return data_types

    def validate_data_types(self, df, db_name, collection_name):
        # Get the data type schema of the MongoDB collection
        collection_data_types = self.get_collection_data_types(db_name, collection_name)

        for col, expected_dtype in collection_data_types.items():
            if col in df.columns:
                try:
                    if expected_dtype in [int, float] and df[col].dtype == object:
                        df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='raise')

                    df[col] = df[col].astype(expected_dtype)

                except (ValueError, pd.errors.OutOfBoundsDatetime) as e:
                    raise ValueError(f"Data type mismatch for column '{col}': {e}")

    def insert_unique_rows(self, df, db_name=None, collection_name=None):
        if db_name:
            self.create_database(db_name)
        if collection_name:
            self.create_collection(collection_name)

        self.validate_data_types(df, db_name, collection_name)

        existing_data = self.find_all(db_name, collection_name)

        if df.empty:
            print("Error: The DataFrame is empty. No rows to insert.")
            return []

        is_collection_empty = existing_data.empty

        if is_collection_empty:
            try:
                df['row_embedding'] = df.apply(row_descriptor.compute_embedding, axis=1)
                return self.insert_many(df.to_dict(orient='records'), db_name, collection_name)
            except Exception as e:
                print(f"Error during MongoDB insertion: {e}")
                return []

        new_data_to_insert = self.filter_duplicate_rows(df, existing_data)

        # Check if there are new rows to insert
        if not new_data_to_insert.empty:
            try:
                new_data_to_insert['row_embedding'] = new_data_to_insert.apply(row_descriptor.compute_embedding, axis=1)
                result = self.insert_many(new_data_to_insert.to_dict(orient='records'), db_name, collection_name)
                return result
            except Exception as e:
                print(f"Error during MongoDB insertion: {e}")
                return []

        print("No new rows to insert. All data already exists in the collection.")
        return []

    def filter_duplicate_rows(self, new_df, existing_df):

        existing_df_cleaned = existing_df.drop(['_id', 'row_embedding'], axis=1, errors='ignore')

        merged_df = pd.merge(new_df, existing_df_cleaned, how='left', indicator=True)

        filtered_df = merged_df[merged_df['_merge'] == 'left_only'].drop('_merge', axis=1)

        return filtered_df









       