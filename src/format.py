import pandas as pd
import numpy as np

class AccidentDataFormatter:
    def __init__(self, 
                 data, 
                 features_int: list = ["obs", "obsm", "choc", "obs2", "obsm2", "choc2", 
                                       "manv", "grav", "situ", "infra", "surf", "atm", 
                                       "col", "catr", "circ", "plan", "prof", "vosp"],
                features_to_keep=["prof","plan","vosp","grav","lum","dep","com","int",
                                  "date","lat","long","atm","circ","nbv","larrout",
                                  "vma","surf","catv","Num_Acc"]
                ):
        self.data = data
        if not data._is_prepared:
            self.data.preprocess_df()

        self.df = self.data.df_final.copy(deep=True)
        self.df_formatted = self.data.df_final

        self.features_int = features_int
        self.features_to_keep = features_to_keep

    def head(self, n: int = 5) -> pd.DataFrame:
        return self.df_formatted.head(n)

    def format(self) -> pd.DataFrame:
        self._int_format()
        self._format_lum()
        self._format_int()
        self._format_prof()
        self._format_atm()
        self._format_circ()
        self._format_plan()
        self._format_vosp()
        self._format_surf()
        self._format_date()
        self._format_agg()
        self._format_nbv()
        self._format_lat_long()
        self._format_com()
        self.df_formatted = self.df_formatted.reset_index(drop=True)[self.features_to_keep]

    def _int_format(self) -> None:
        for f in self.features_int:
            if f[-1] == "2" and self.data.second_vehicule:
                self.df_formatted[f] = self.df_formatted[f].replace(-1, np.nan)
                self.df_formatted[f] = self.df_formatted[f].astype("Int64")

    def _format_lum(self) -> None:
        self.df_formatted["lum"] = self.df_formatted["lum"].replace({1: 0, 2: 1, 5: 1, 3: 2, 4: 2})

    def _format_int(self) -> None:
        self.df_formatted["int"] = self.df_formatted["int"].replace(
            {-1: np.nan, 0: np.nan, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 3, 9: 3}
        )

    def _format_prof(self) -> None:
        self.df_formatted["prof"] = self.df_formatted["prof"].replace(
            {
                -1: np.nan,
                0: 1,
            }
        )

    def _format_atm(self) -> None:
        self.df_formatted["atm"] = self.df_formatted["atm"].replace(
            {-1: np.nan, 1: 0, 2: 1, 8: 1, 7: 2, 5: 2, 3: 3, 4: 3, 6: 3, 9: 3}
        )

    def _format_circ(self) -> None:
        self.df_formatted["circ"] = self.df_formatted["circ"].replace({-1: np.nan, 1: 0, 2: 1, 3: 2, 4: 3})

    def _format_plan(self) -> None:
        self.df_formatted["plan"] = self.df_formatted["plan"].replace(
            {
                -1: np.nan,
                1: 0,
                2: 1,
                3: 1,
                4: 2,
            }
        )

    def _format_vosp(self) -> None:
        self.df_formatted["vosp"] = self.df_formatted["vosp"].replace({-1: np.nan, 0: 0, 3: 1, 2: 2, 1: 3})

    def _format_surf(self) -> None:
        self.df_formatted["surf"] = self.df_formatted["surf"].replace(
            {-1: np.nan, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 3, 9: 3}
        )

    def _format_date(self) -> None:
        self.df_formatted["an"] = self.df_formatted["an"].apply(
            lambda x: int(x) + 2000 if int(x) < 100 else int(x)
        )
        self.df_formatted["date"] = (
            self.df_formatted["jour"].astype(str)
            + "/"
            + self.df_formatted["mois"].astype(str)
            + "/"
            + self.df_formatted["an"].astype(str)
        )
        self.df_formatted["date"] = pd.to_datetime(self.df_formatted["date"], format="%d/%m/%Y")
        self.df_formatted["date"] = self.df_formatted["date"].dt.date
        self.df_formatted = self.df_formatted.drop(columns=["jour", "mois", "an"])
    
    def _format_agg(self) -> None:
        self.df_formatted["is_in_agg"] = self.df_formatted["agg"].replace({1: 0, 2: 1})
        self.df_formatted = self.df_formatted.drop(columns=["agg"])

    def _format_nbv(self) -> None:
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
        self.df_formatted["nbv"] = self.df_formatted["nbv"].replace(-1, np.nan)
        self.df_formatted["nbv"] = self.df_formatted["nbv"].astype("Int64")
        self.df_formatted = remove_outliers_with_nan(self.df_formatted, "nbv")
    
    def _format_lat_long(self) -> None:
        def apply_format_lat_long(row):
            nord = 52
            sud = 41.1
            est = 9.56
            ouest = -4.8
            if str(row[0]) != "-" and str(row[1]) != "-":
                lat = float(str(row[0]).replace(',', '.')) if str(row[0])[0] != "-" else float(str(row[0])[1:].replace(',', '.')) * -1
                lon = float(str(row[1]).replace(',', '.')) if str(row[1])[0] != "-" else float(str(row[1])[1:].replace(',', '.')) * -1
            else:
                lat = np.nan
                lon = np.nan
            i = 0
            while lat > nord:
                i += 1
                lat = lat / 10
            
            while lon < ouest or lon > est:
                lon = lon / 10
            
            row[0] = lat
            row[1] = lon
                
            return row
        self.df_formatted[["lat", "long"]] = self.df_formatted[["lat", "long"]].apply(apply_format_lat_long, axis=1)

    def _format_com(self) -> None:
        def try_convert(a):
            a = str(a)
            try:
                return int(a[:2])
            except:
                return np.nan
        self.df_formatted['com'] = self.df_formatted['com'].apply(lambda a : try_convert(a))
    
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
    df = df[df['is_in_agg'] == 1]
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

def surf(df : pd.DataFrame):
    df['surf'] = df['surf'].replace({1:1,2:2,3:2,4:3,5:4,6:2,7:4,8:5,9:5})
    return df

def obs(df : pd.DataFrame):
    df['obs'] = df['obs'].replace({1:1,2:2,3:3,4:3,5:3,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15,16:16,17:17})
    df['obs2'] = df['obs2'].replace({1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15,16:16,17:17})
    return df

def lat_long(df : pd.DataFrame):

    #création d'un mask pour éliminer les latitutes et longitudes qui sont de len inférireur à 5 => pas précises
    long_mask = df['long'].astype(str).apply(len) >= 5
    lat_mask = df['lat'].astype(str).apply(len) >= 5
    mask = np.logical_and(long_mask, lat_mask)
    df =  df[mask]

    #fonction de formatage pour les lat et long, fonction qui prends les points les  extrems de france et divise par 10 les éléments pour
    #les faire rentrer dans la grille formée par les extremes.
    def format_lat_long(row):
        nord = 52
        sud = 41.1
        est = 9.56
        ouest = -4.8
        lat = float(str(row[0]).replace(',', '.'))
        lon = float(str(row[1]).replace(',', '.'))
        i = 0
        while lat > nord:
            i += 1
            lat = lat / 10
        
        while lon < ouest or lon > est:
            lon = lon / 10
        
        row[0] = lat
        row[1] = lon
            
        return row
    #applique la fonction de fromattage sur l'ensemble du dataset
    df[["lat", "long"]] = df[["lat", "long"]].apply(format_lat_long, axis=1)
    return df

def com(df : pd.DataFrame):
    
    def try_convert(a):
        a = str(a)
        try:
            return int(a[:2])
        except:
            return np.nan

    df['com'] = df['com'].apply(lambda a : try_convert(a))
    df = df.dropna(subset=['com'], axis=0)
    return df

def drop_features(df : pd.DataFrame):

    df = df.drop(['Num_Acc', 'manv', 'manv2', 'larrout', 'vma', 'num_veh',
                  'v1', 'v2', 'voie', 'motor', 'id_vehicule', 
                  'pr', 'pr1', 'lartpc', 'hrmn', 'Unnamed: 0', 'occutc', 'env1', 'gps'], axis=1)
    return df

def merge_post_format(dataframe : pd.DataFrame, formating_list : list = None):

    if formating_list == None:
        formating_list = [features, date, lum, agg, int_feature, nbv, catv, lat_long, com, drop_features]
    df = dataframe.copy()
    for f in formating_list:
        df = df.copy()
        df = f(df)

    return df.reset_index(drop=True)
