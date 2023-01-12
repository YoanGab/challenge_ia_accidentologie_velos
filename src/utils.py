from colorama import Fore
import pandas as pd
import os
from format import merge_post_format

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

def load_accident(filename="dataset_velo_acc_preprocess.csv", path : list[str] = ['..', 'data'], format=True):

    df = pd.read_csv(os.path.join(*path, "dataset_velo_acc_preprocess.csv"))

    if format == True:
        df = merge_post_format(df)
    
    return df
    

