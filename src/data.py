import pandas as pd
import numpy as np
from src.data import AccidentData

class AccidentDataFormatter:
    def __init__(self, 
                 data, 
                 merged: bool = False,
                 features_int: list = ["obs", "obsm", "choc", "obs2", "obsm2", "choc2", 
                                       "manv", "grav", "situ", "infra", "surf", "atm", 
                                       "col", "catr", "circ", "plan", "prof", "vosp"],
                features_to_keep=["prof","plan","vosp","grav","lum","dep","com","int",
                                  "date","lat","long","atm","circ","nbv","larrout",
                                  "vma","surf","catv","Num_Acc"]
                ):
        self.data = data
        #if not data._is_prepared:
        #    self.data.preprocess_df()

        if merged:
            self.df = self.data.df_merged.copy(deep=True)
            self.df_formatted = self.data.df_merged
        else:
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
            elif f[-1] != "2":
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
        #self.df_formatted = remove_outliers_with_nan(self.df_formatted, "nbv")
    
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
    
