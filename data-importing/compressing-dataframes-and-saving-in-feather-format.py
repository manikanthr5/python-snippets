"""
Credits: https://www.kaggle.com/corochann/ashrae-feather-format-for-fast-loading
"""

import pandas as pd
import numpy as np
import feather
import warnings
warnings.filterwarnings('ignore')
from pandas.api.types import is_datetime64_any_dtype as is_datetime

def reduce_mem_usage(df, use_float16=False):
    """ Iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        if is_datetime(df[col]):
            # skip datetime type
            continue
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def import_data(file, datetime_cols=[]):
    """ Import a dataframe and optimize its memory usage"""
    print('-'*50)
    print('Processing', file)
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    for col in datetime_cols:
        try:
            df[col] = pd.to_datetime(df[col])
        except:
            print(col, 'is not present in', file)
    df = reduce_mem_usage(df)
    return df

def main():
    # Import and Compress dataframes.
    df_train = import_data('../data/train.csv', datetime_cols=['timestamp'])
    df_test = import_data('../data/test.csv', datetime_cols=['timestamp'])

    # Save compressed dataframes to feather format.
    df_train.to_feather('../data/train.feather')
    df_test.to_feather('../data/test.feather')

    # Import dataframes from feather format.
    df_train = feather.read_dataframe('../data/train.feather')
    df_test = feather.read_dataframe('../data/test.feather')

if __name__ == "__main__":
    main()
