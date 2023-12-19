import pandas as pd

def troin_sales(file):
    df = pd.read_excel(file)
    df.drop(['Ref', 'Local DR'], axis = 1, inplace=True)
    df.dropna(inplace=True)

    return df