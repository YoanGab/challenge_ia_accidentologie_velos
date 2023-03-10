from colorama import Fore
import pandas as pd
import os
from format import merge_post_format
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
import numpy as np
from shapely.geometry import shape
import geopandas as gpd

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
    df_encoded = pd.DataFrame(df_encoded, columns=encoder.get_feature_names_out(categorical_cols))
    df = df.drop(categorical_cols, axis=1)
    df = pd.concat([df, df_encoded], axis=1)
    return df

def impute(df, feature ):
    imp = KNNImputer(missing_values = np.nan, n_neighbors=1)
    series_date = df["date"]
    series_adr = df["adr"]
    series_gps = df["gps"]
    df = pd.DataFrame(imp.fit_transform(df.drop(columns=["date", "adr", "gps"], axis=1)))
    df = pd.concat([df, series_date])
    df = pd.concat([df, series_adr])
    df = pd.concat([df, series_gps])
    return df




def load_accident__do_not_use(filename="dataset_velo_acc_preprocess.csv", path : list[str] = ['..', 'data'], processed=True):
    """
    DO NOT USE
    """
    df = pd.read_csv(os.path.join(*path, "dataset_velo_acc_preprocess.csv"))

    if processed == True:
        df = merge_post_format(df)
        df = df.replace(pd.NA, np.nan)
        df = one_hot_encoder(df, ['lum', 'int', 'atm', 'col', 'catr', 'circ', 'vosp','prof', 'plan', 'surf', 'infra', 'situ', 'catv', 'obs', 'obsm', 'choc','grav', 'obs2', 'obsm2', 'choc2'])
        #il faut standarScaler les features num??rique mais y'en a pas beaucoup (genre 1 ou 2)
        df = impute(df)

    return df
    

def create_lat_long(df):
    """
    Replace line by 4 float (Lat1, Long1, Lat2, Long2) into the whole dataframe
    """

    def get_coords(line):
        """
        Transform a line type into 4 float (Lat1, Long1, Lat2, Long2)
        """
        # On cr??e un objet g??om??trique ?? partir de la linestring
        line = shape(line)
        # On r??cup??re les coordonn??es du premier point de la linestring
        longitude_dep, latitude_dep = line.coords[0]
        longitude_fin, latitude_fin = line.coords[1]
        return latitude_dep, longitude_dep, latitude_fin, longitude_fin

    df["coords"] = df["geometry"].apply(get_coords)
    df[["latitude_dep", "longitude_dep", "latitude_fin", "longitude_fin"]] = pd.DataFrame(df["coords"].tolist(), index=df.index)
    df = df.drop("coords", axis=1)
    return df

def infra_preprocessing(df):
    """
    Drop useless columns (too much nans) and drop some nan lines.
    """
    #hesite to drop : regime_d, code_com_d, regime_g, code_come_g
    df = create_lat_long(df) #Transform line to point

    def fill_missing_values(row, f):
        if row[f + "_d"] == None or row[f + "_d"] == "AUCUN":
            row[f + "_d"] = row[f + '_g']
        return row

    def try_convert(a):
        a = str(a)
        try:
            return int(a[:2])
        except:
            return np.nan

    df['code_com_d'] = df['code_com_d'].apply(lambda a : try_convert(a))
    df['code_com_g'] = df['code_com_g'].apply(lambda a : try_convert(a))
    df = df.apply(fill_missing_values, axis=1, args=("code_com",))

    return df

    """
    df = df.apply(fill_missing_values, axis=1, args=("ame",))
    df = df.apply(fill_missing_values, axis=1, args=("regime",))

    df = df[["code_com_d", "ame_d", "regime_d","date_maj", "latitude_dep", "longitude_dep", "latitude_fin", "longitude_fin"]]
    df.columns = ["code_com", "ame", "regime","date_maj", "latitude_dep", "longitude_dep", "latitude_fin", "longitude_fin"] """

    #df = df.dropna(subset=["code_com"])
    df = df.reset_index()
    return df

def load_infra():
    df = gpd.read_file("../data/france-20230101-formatted.geojson")
    df = infra_preprocessing(df)
    
    return df


def statistic_dataframe(df):
    day_mean = df.groupby(['id_compteur'])['sum_by_day'].mean().astype('int32')
    month_mean = df.groupby(['id_compteur'])['sum_by_month'].mean().astype('int32')
    year_mean = df.groupby(['id_compteur'])['sum_by_year'].mean().astype('int32')
    
    new_df = pd.merge(day_mean, month_mean, on=['id_compteur'], how='left') 
    new_df = pd.merge(new_df, year_mean, on=['id_compteur'], how='left') 
    
    
    lat = df.groupby(['id_compteur'])['latitude'].mean() #same value for all value, it doesn't change anything
    long = df.groupby(['id_compteur'])['longitude'].mean()
    
    new_df = pd.merge(new_df, lat, on=['id_compteur'], how='left')
    new_df = pd.merge(new_df, long, on=['id_compteur'], how='left')
    
    adresse = df.groupby(['id_compteur'])['name'].unique().map(lambda x: x[0])
    new_df = pd.merge(new_df, adresse, on=['id_compteur'], how='left')
    
    date_install = df.groupby(['id_compteur'])['installation_date'].unique().map(lambda x: x[0])
    new_df = pd.merge(new_df, date_install, on=['id_compteur'], how='left')
        
    new_df.rename(columns = {'sum_by_day':'mean_day', 'sum_by_month':'mean_month', 'sum_by_year':'mean_year', 'name':'adresse'}, inplace = True)
    
    return new_df

def comptage_by_date(df):
    df['day'] = df['date'].str[:10] 
    
    #On fait la somme des velos compt??s par jour ?? chaque point de comptage
    result = df.groupby(['id_compteur', 'day'])['sum_counts'].sum()
    
    #Ajout des r??sultats dans une nouvelle colonne afin de garder l'ensemble des informations initiales
    df = pd.merge(df, result, on=['id_compteur', 'day'], how='left') 
    df.rename(columns = {'sum_counts_x':'sum_counts', 'sum_counts_y':'sum_by_day'}, inplace = True)
    
    df['month'] = df['date'].str[:7] 
    
    #On fait la somme des velos compt??s par mois ?? chaque point de comptage
    result = df.groupby(['id_compteur', 'month'])['sum_counts'].sum()
    
    #Ajout des r??sultats dans une nouvelle colonne afin de garder l'ensemble des informations initiales
    df = pd.merge(df, result, on=['id_compteur', 'month'], how='left') 
    df.rename(columns = {'sum_counts_x':'sum_counts', 'sum_counts_y':'sum_by_month'}, inplace = True)
    
    df['year'] = df['date'].str[:4] 
    
    #On fait la somme des velos compt??s par ann??e ?? chaque point de comptage
    result = df.groupby(['id_compteur', 'year'])['sum_counts'].sum()
    
    #Ajout des r??sultats dans une nouvelle colonne afin de garder l'ensemble des informations initiales
    df = pd.merge(df, result, on=['id_compteur', 'year'], how='left') 
    df.rename(columns = {'sum_counts_x':'sum_counts', 'sum_counts_y':'sum_by_year'}, inplace = True)
    
    return df

def handle_coordinates(df):
    #format colonne coordinates : 'latitude,longitude'
    df['latitude'] = df['coordinates'].apply(lambda x: x.split(',')[0]).astype('float')
    df['longitude'] = df['coordinates'].apply(lambda x: x.split(',')[1]).astype('float')
    
    return df

def load_accident():

    df = pd.read_csv(os.path.join("..", "data", "dataset_velo_acc_preprocess.csv"))
    df = merge_post_format(df)
    df = df.replace(pd.NA, np.nan)


    #df.dropna(subset=['lat', 'long'], axis=0, inplace=True) 
    return df


def closest_point(A, B, C):
    """
    Point le plus proche de A appartenant ?? la droit BC
    """
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    AB = C - B 
    AC = A - B
    dot = np.dot(AB, AC)
    proj = dot / np.dot(AB, AB)
    P = B + proj * AB
    return P

def distance_infra_acc(raw, lat_acc, long_acc):
    """ 
    Distance entre l'infrastucture raw et le point d'accident lat_acc, long_acc
    """
    point = closest_point([lat_acc,long_acc], [raw['latitude_dep'], raw['longitude_dep']], [raw['latitude_fin'], raw['longitude_fin']])
    dist = great_circle((lat_acc,long_acc), point).meters
    return dist