from colorama import Fore
import pandas as pd
import os
from format import merge_post_format
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
import numpy as np

def missing(df, detail=True):
    total = 0
    for col in df.columns:
        miss = df[col].isnull().sum()
        pct = df[col].isna().mean() * 100
        total += miss
        if miss != 0:
            if pct>10: color=Fore.RED
            else: color=Fore.YELLOW
            print(color+'{} => {} [{}%]'.format(col, miss, round(pct, 2)))
        
        elif (total == 0) and(detail):
            print(Fore.GREEN+'{} => no missing values [{}%]'.format(col, 0))
        total=0

def one_hot_encoder(df, categorical_cols):
    encoder = OneHotEncoder()
    df_encoded = encoder.fit_transform(df[categorical_cols]).toarray()
    df_encoded = pd.DataFrame(df_encoded, columns=encoder.get_feature_names(categorical_cols))
    df = df.drop(categorical_cols, axis=1)
    df = pd.concat([df, df_encoded], axis=1)
    return df

def impute(df):

    imp = KNNImputer(missing_values = np.nan, n_neighbors=1)
    series_date = df["date"]
    df = pd.DataFrame(imp.fit_transform(df.drop('date', axis=1)))
    return pd.concat([df, series_date])

def load_accident(filename="dataset_velo_acc_preprocess.csv", path : list[str] = ['..', 'data'], processed=True):

    df = pd.read_csv(os.path.join(*path, "dataset_velo_acc_preprocess.csv"))

    if processed == True:
        df = merge_post_format(df)
        df = df.replace(pd.NA, np.nan)
        df = one_hot_encoder(df, ['lum', 'int', 'atm', 'col', 'catr', 'circ', 'vosp','prof', 'plan', 'surf', 'infra', 'situ', 'catv', 'obs', 'obsm', 'choc','grav', 'obs2', 'obsm2', 'choc2'])
        #il faut standarScaler les features numérique mais y'en a pas beaucoup (genre 1 ou 2)
        df = impute(df)

    return df
    

