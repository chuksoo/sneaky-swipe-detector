import pandas as pd


def get_data(url):
    try:
        data = pd.read_json(url, compression='zip', lines=True)
        data.to_parquet('../data/raw/transactions.parquet", engine="fastparquet')
    except:
        data = pd.read_parquet('../data/raw/sweaky-swipe-transactions.parquet')
    return data
