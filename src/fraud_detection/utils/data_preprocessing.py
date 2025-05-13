import pandas as pd
from scripts.data_download import get_data

class PreprocessData:
    def __init__(self, url: str):
        ''' Download raw transaction data from url and store as pandas dataframe'''
        self.url = url
        self.df = get_data(url)

    def change_datatype(self, cols: list[str], dtype: str) -> None:
        '''Convert the columns in col to the given datatype and modifies self.df in place'''
        for col in cols:
            self.df[col] = self.df[col].astype(dtype)

    def create_features(self) -> None:
        '''Feature engineer new features from datetime features'''
        df = self.df
        df['transactionMonth'] = df['transactionDateTime'].dt.month_name()
        df['transactionDayofWeek'] = df['transactionDateTime'].dt.day_name()
        df['transactionHour'] = df['transactionDateTime'].dt.hour.astype("int32")
        df['transactionMinutes'] = df['transactionDateTime'].dt.minute.astype("int32")
        df['transactionSeconds'] = df['transactionDateTime'].dt.second.astype("int32")
        df['currentExpMonth'] = df['currentExpDate'].dt.month_name()
        df['currentExpDayofWeek'] = df['currentExpDate'].dt.day_name()
        df['accountOpenMonth'] = df['accountOpenDate'].dt.month_name()
        df['accountOpenDayofWeek'] = df['accountOpenDate'].dt.day_name()
        df['dateOfLastAddressChangeMonth'] = df['dateOfLastAddressChange'].dt.month_name()
        df['dateOfLastAddressChangeDayofWeek'] = df['dateOfLastAddressChange'].dt.day_name()
        self.df = df

    def filter_duplicates(self, window_seconds: int = 120) -> None:
        """
        Mark any two consecutive transactions for the same customer/amount,
        within `window_seconds`, as duplicates, then drop them.
        """
        self.df = self.df.sort_values('transactionDateTime')
        # Compute the time difference (in seconds) between consecutive rows per (customerId, transactionAmount)
        self.df['timeDiff'] = (
            self.df.groupby(['customerId', 'transactionAmount'], sort=False)
              ['transactionDateTime']
              .diff()
              .dt.total_seconds()
        )
        # Drop any row whose timeDiff <= window_seconds since it's considered a duplicate
        self.df['isDuplicated'] = self.df['timeDiff'].le(window_seconds)
        self.df = self.df[~self.df['isDuplicated']].drop(columns=['timeDiff', 'isDuplicated'])

    def feature_pipeline(self) -> pd.DataFrame:
        """
        Feature pipeline consisting of:
            - Ingestion and type casting
            - Feature Engineering
            - Filtering and deduplication
        Return the cleaned DataFrame.
        """
        # --- ingest + type cast ---
        self.change_datatype(
            ['transactionDateTime', 'currentExpDate', 
             'accountOpenDate', 'dateOfLastAddressChange'],
            'datetime64[ns]'
        )
        self.change_datatype(
            ['accountNumber', 'customerId', 'creditLimit',
             'cardCVV', 'enteredCVV', 'cardLast4Digits'],
            'int32'
        )
        self.change_datatype(
            ['availableMoney', 'transactionAmount', 'currentBalance'],
            'float32'
        )

        # --- feature engineering ---
        self.create_features()

        # --- dedupe ---
        self.filter_duplicates(window_seconds=120)

        return self.df   
    
    def save_parquet(self, path: str) -> None:
        self.df.to_parquet(path, index=False)


if __name__ == "__main__":
    url = 'https://github.com/CapitalOneRecruiting/DS/blob/master/transactions.zip?raw=true'
    processor = PreprocessData(url)
    df = processor.feature_pipeline()
    processor.save_parquet('../data/processed/transactions_clean.parquet', engine='fastparquet')
    print(df.head())
    print(df.dtypes)


