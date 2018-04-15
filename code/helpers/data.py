import pandas as pd
import numpy as np


def get_data_raw_model(path):
    # Load DF
    df = pd.read_csv(path, sep='\t')

    # Remove some bad variables
    df = df[(df['CHOICE'] != 0) & (df['CAR_TT'] > 0) & (df['AGE'] < 6)]

    # Change some columns
    df.loc[:, 'TRAIN_CO'] = df['TRAIN_CO'] * (df['GA'] == 0).astype(int)
    df.loc[:, 'SM_CO'] = df['SM_CO'] * (df['GA'] == 0).astype(int)
    df['SENIOR'] = (df['AGE'] == 5).astype(int)

    cols = ['CHOICE', 'TRAIN_TT', 'TRAIN_CO', 'SM_TT', 'SM_CO', 'CAR_TT', 'CAR_CO', 'SENIOR', 'TRAIN_HE', 'SM_HE']
    df = df[cols]

    # cols_to_norm = ['TRAIN_TT', 'TRAIN_CO', 'SM_TT', 'SM_CO', 'CAR_TT', 'CAR_CO', 'TRAIN_HE', 'SM_HE']

    # Normalize the values
    # for c in cols_to_norm:
    #     df[c] = df[c] / np.max(df[c])

    df.index = range(len(df))

    return df


def get_data_norm_model(path):
    # Load DF
    df = pd.read_csv(path, sep='\t')

    # Remove some bad variables
    df = df[(df['CHOICE'] != 0) & (df['CAR_TT'] > 0) & (df['AGE'] < 6)]

    # Change some columns
    df.loc[:, 'TRAIN_CO'] = df['TRAIN_CO'] * (df['GA'] == 0).astype(int)
    df.loc[:, 'SM_CO'] = df['SM_CO'] * (df['GA'] == 0).astype(int)
    df['SENIOR'] = (df['AGE'] == 5).astype(int)

    cols = ['CHOICE', 'TRAIN_TT', 'TRAIN_CO', 'SM_TT', 'SM_CO', 'CAR_TT', 'CAR_CO', 'SENIOR', 'TRAIN_HE', 'SM_HE']
    df = df[cols]

    cols_to_norm = ['TRAIN_TT', 'TRAIN_CO', 'SM_TT', 'SM_CO', 'CAR_TT', 'CAR_CO', 'TRAIN_HE', 'SM_HE']

    # Normalize the values
    for c in cols_to_norm:
        df[c] = df[c] / 100

    df.index = range(len(df))

    return df