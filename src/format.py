import pandas as pd
import numpy as np

def features(df):

    features = ["obs", "obsm", "choc", "obs2", "obsm2", "choc2", "manv", "grav", "situ", "infra", "surf", "atm", "col", "catr", "circ", "plan", "prof", "vosp"]
    
    for f in features:  
        df[f] = df[f].replace(-1, np.nan)
        df[f] = df[f].astype("Int64")
    return df

def date(df: pd.DataFrame) -> pd.DataFrame:
    df["an"] = df["an"].apply(lambda x: int(x) + 2000 if int(x) < 100 else int(x))
    df["date"] = df["jour"].astype(str) + "/" + df["mois"].astype(str) + "/" + df["an"].astype(str)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    df["date"] = df["date"].dt.date
    df = df.drop(columns=["jour", "mois", "an"])
    return df

def lum(df: pd.DataFrame) -> pd.DataFrame:
    df["lum"] = df["lum"].replace({1: 1, 2: 2, 3: 5, 4: 4, 5: 3})
    return df

def agg(df: pd.DataFrame) -> pd.DataFrame:
    df["is_in_agg"] = df["agg"].replace({1: 0, 2: 1})
    df = df.drop(columns=["agg"])
    return df

def int_feature(df: pd.DataFrame) -> pd.DataFrame:
    df["int"] = df["int"].replace({0: np.nan, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8}).astype("Int64")
    return df
    
def nbv(df : pd.DataFrame) -> pd.DataFrame:
    def remove_outliers_with_nan(df, column):
        df2 = df[np.isfinite(df[column])]
        df1 = df[df[column].isna()]
        q1 = df2[column].quantile(0.25)
        q3 = df2[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (2.5 * iqr)
        upper_bound = q3 + (2.5 * iqr)
        df2 = df2[(df2[column] > lower_bound) & (df2[column] < upper_bound)]
        return pd.concat([df1, df2], axis=0)
        
    df["nbv"] = df["nbv"].replace(-1, np.nan)
    df["nbv"] = df["nbv"].astype("Int64")
    df = remove_outliers_with_nan(df, "nbv")  
    return df


def catv(df : pd.DataFrame):
    f = 'infra'
    df[f] = df[f].replace(-1, np.nan) #replace -1 (non renseigné) with np.nan values
    df[f] = df[f].astype('Int64')

    def merge_vehicule(row):
        if row['catv'] == 1:
            row['catv'] = row['catv2']
        elif type(row['catv']) ==  pd._libs.missing.NAType:
            row['catv'] == row['catv2']
        #else (if row catv != 1) okay
        return row

    to_drop = 'catv2'
    catv_dict = {0:np.nan,1:1,2:2,3:3,4:2,5:2,6:2,7:3,8:3,9:3,10:3,11:3,12:3,13:4,14:4,15:4,16:4,17:4,18:5,19:6,20:7,21:7,30:2,31:2,32:2,33:2,34:2,35:3,36:3,37:5,38:5,39:6,40:6,41:3,42:3,43:3,50:1,60:1,80:1,99:7}
    df = df.apply(merge_vehicule, axis=1)
    df['catv'] = df['catv'].replace(catv_dict)
    df = df.drop(to_drop, axis=1)
    return df

"""def surf(df : pd.DataFrame):
    df['surf'] = df['surf'].replace({1:1,2:2,3:2,4:3,5:4,6:2,7:4,8:5,9:5})
    return df"""

"""def obs(df : pd.DataFrame):
    df['obs'] = df['obs'].replace({1:1,2:2,3:3,4:3,5:3,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15,16:16,17:17})
    df['obs2'] = df['obs2'].replace({1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15,16:16,17:17})
    return df"""

def drop_features(df : pd.DataFrame):

    df = df.drop(['Num_Acc', 'manv', 'manv2', 'larrout', 'vma'], axis=1)
    return df


def merge_post_format(dataframe : pd.DataFrame, formating_list : list = None):

    if formating_list == None:
        formating_list = [features, date, lum, agg, int_feature, nbv, catv, surf, drop_features]
    df = dataframe.copy()
    for f in formating_list:
        df = df.copy()
        df = f(df)

    return df.reset_index(drop=True)