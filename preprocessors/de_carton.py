import tabula
import re

import pandas as pd

def convert_columns_with_commas_to_numeric(df):
    for col in df.columns:
        if df[col].dtype == 'object':  # Check only object (string) columns
            if df[col].apply(lambda x: isinstance(x, str) and re.match(r'^-?\d{1,3}(,\d{3})*(\.\d+)?$', x)):
                # If the column contains strings with commas, convert to numeric
                df[col] = df[col].replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce')
    return df

def preprocess_monthly_ap(file_path, area=[80.0,35.0,210.68,734.93]):
    df_AP = tabula.read_pdf(file_path, area=area)
    df = df_AP[0]
    for i in range(len(df)-1, 0, -1):
        if df.iloc[i, 1:].isna().all():
            df.at[i - 1, 'Creditor'] += ' ' + df.at[i, 'Creditor']
            df = df.drop(index=i)
    df = df.reset_index(drop=True)
    df = df.set_index('Creditor')
    df = df.T
    date = pd.to_datetime(df.index, format='%b-%Y')
    df['Year'] = date.year
    df['Month'] = date.month_name()
    df = df.reset_index(drop=True)

    for col in df.columns:
        if df[col].dtype == object:  # Check only object (string) columns
            if df[col].str.contains(',').any():  # Check if column contains commas
                df[col] = df[col].replace(',', '', regex=True).apply(pd.to_numeric)

    return df

def preprocess_monthly_ar(file_path):
    df_AR = tabula.read_pdf(file_path, area=[80.0,35.0,210.68,734.93])
    df = df_AR[0]
    for i in range(len(df)-1, 0, -1):
        # Check if all values in the row (except the first column) are NA
        if df.iloc[i, 1:].isna().all():
            # Concatenate the first column with the preceding row
            df.at[i - 1, 'Debtor'] += ' ' + df.at[i, 'Debtor']
            # Drop the current row
            df = df.drop(index=i)

    df = df.reset_index(drop=True)
    df = df.set_index('Debtor')
    df = df.T
    date = pd.to_datetime(df.index, format='%b-%Y')
    df['Year'] = date.year
    df['Month'] = date.month_name()
    df = df.reset_index(drop=True)

    return df

if __name__ == '__main__':
    df = preprocess_monthly_ap('data/de_carton/AP MONTHLY PURCHASE ANALYSIS REPORT 08-12-2023.pdf')
    print(df)
    df.to_csv('data/de_carton/AP MONTHLY PURCHASE ANALYSIS REPORT 08-12-2023.csv')