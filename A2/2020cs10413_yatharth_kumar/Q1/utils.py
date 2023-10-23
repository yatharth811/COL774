import pandas as pd
def read_data(path: str):
    reader = pd.read_csv(path)
    return reader

def lower(df):
  df['CoronaTweet'] = df['CoronaTweet'].str.lower()
  return df