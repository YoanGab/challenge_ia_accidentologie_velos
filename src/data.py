
import pandas as pd
import json
import requests
import os
from tqdm.notebook import tqdm
import shutil

class AccidentData:
    def __init__(self, load=False):
        # Répertoire où se trouvent les données
        self.data_dir = os.path.join("..","data")
        # Liste des catégories de données disponibles
        self.categories = ["usagers", "vehicules", "lieux", "caracteristiques"]
        # Lecture des URLs à partir du fichier JSON
        self.get_json_files()
        # Années pour lesquelles il y a des données disponibles
        self.years = list(self.urls["usagers"].keys())
        # Vérification et contrôle des données
        self.check_and_control_data()
        self.df_final = None
        self._is_prepared = False
    
    def head(self, n: int = 5) -> pd.DataFrame:
        if not self._is_prepared:
            self.preprocess_df()
        return self.df_final.head(n)
    
    def get_json_files(self):
        """
        Charge les URLs des données dans l'attribut self.urls.
        """
        with open(os.path.join(self.data_dir, "accident_corporels_urls.json"),"r") as file:
            self.urls = json.load(file)
        with open(os.path.join(self.data_dir, "description_features.json"),"r", encoding="utf-8") as file:
            self.feat_desc = json.load(file)
    
    def download_data(self):
        """
        Télécharge les données manquantes.
        """
        # S'il y a des fichiers manquants
        if len(self.filenames) > 0:
            print("[Check] Checking completed, some data is missing!")
            print("[Download] Downloading missing data...")
            # Pour chaque catégorie de données
            for filename in self.filenames.keys():
                filename_dir = os.path.join(self.data_dir,filename) # Répertoire où stocker les données
                print(f"\n[Download] Downloading {filename} files...")
                # Pour chaque année
                for i, year in enumerate(self.filenames[filename]):
                    # Téléchargement des données
                    data = requests.get(self.urls[filename][year]).text

                     # Création du répertoire s'il n'existe pas
                    if not os.path.exists(filename_dir):
                        os.makedirs(filename_dir)

                    # Écriture des données dans un fichier CSV
                    with open(os.path.join(filename_dir,f"{year}.csv"), 'w', encoding='utf-8') as f:
                        f.write(data)

                    # Affichage de la barre de progression
                    bar_length = int(50 * (i+1) / len(self.filenames[filename]))
                    bar = "#" * bar_length + "-" * (50 - bar_length)
                    print(f"{i+1}/{len(self.filenames[filename])} [{bar}]", end='\r')
            print("\n[Download] Download completed!")
        # S'il n'y a pas de fichiers manquants
        else:
            print("[Check] Checking completed, no data is missing!")

    def check_missing_data(self):
        """
        Vérifie les données manquantes.
        """
        print("[Check] Checking if data is in your computer...")
        # Dictionnaire des fichiers manquants par catégorie
        self.filenames = {}
        # Pour chaque catégorie de données
        for categorie in self.categories:
            # Chemin du répertoire de cette catégorie
            filename_path = os.path.join(self.data_dir,categorie)
            # Si le répertoire de cette catégorie n'existe pas, tous les fichiers de cette catégorie sont manquants
            if not os.path.exists(filename_path):
                self.filenames[categorie] = self.years
            # Si le répertoire existe, vérification des fichiers manquants
            else:
                for year in self.years:
                    # Si le fichier de cette année n'existe pas, il est considéré comme manquant
                    if not os.path.exists(os.path.join(filename_path, f"{year}.csv")):
                        # Ajout de l'année au dictionnaire des fichiers manquants
                        if categorie in self.filenames:
                            self.filenames[categorie].append(year)
                        else:
                            self.filenames[categorie] = [year]
                            
    def check_and_control_data(self):
        """
        Vérifie les données manquantes et les télécharge si nécessaire.
        """
        self.check_missing_data()
        self.download_data()
    
    def reset_db(self):
        """
        Réinitialise les données en supprimant tous les répertoires de données et en vérifiant les données manquantes.
        """
        print("[Reset] Reseting data...")
        # Suppression de tous les répertoires de données
        for categorie in self.categories:
            categorie_path = os.path.join(self.data_dir, categorie)
            if os.path.exists(categorie_path):
                shutil.rmtree(categorie_path, ignore_errors=True) 
        print("[Reset] Data have been deleted")
        # Téléchargement des données
        self.check_and_control_data()
        print("[Reset] Data have been reset")

    def get_pd_file_from_year(self, cat, begin, end=None, merge=True):
        """
        Récupère un DataFrame pandas à partir de l'année demandée.
        
        Parameters:
        cat (str) : Catégorie de données à récupérer.
        begin (int) : Année de début.
        end (int) : Année de fin (optionnel, par défaut None).
        merge (bool) : Si True, fusionne les DataFrames de chaque année en un seul DataFrame. Si False, renvoie une liste de DataFrames.
        
        Returns:
        DataFrame pandas ou liste de DataFrames
        """
        # Si la catégorie de données demandée est valide
        if cat.lower() in self.categories:
            cat_path = os.path.join(self.data_dir, cat)
            # Si une seule année est demandée
            if end == None:
                # Si l'année demandée existe dans les données
                if str(begin) in self.years:
                    # Chargement du fichier CSV en tant que DataFrame pandas
                    return pd.read_csv(os.path.join(cat_path, f"{begin}.csv"), sep=None, engine='python')
            # Si une plage d'années est demandée
            else:
                if str(begin) in self.years and str(end) in self.years:
                    list_df = []
                    for annee in range(begin,end+1):
                        list_df.append(pd.read_csv(os.path.join(cat_path, f"{str(annee)}.csv"), sep=None, engine='python'))
                    if merge:
                        return pd.concat(list_df)
                    else:
                        return list_df
        else:
            raise ValueError("La catégorie de données demandée n'est pas valide")
            
    def get_merge_df(self, begin, end, check=False, name="df_merge.csv"):
        name = os.path.join(self.data_dir ,name)
        print("[Check] Checking if the file already exists...")
        if os.path.exists(name) and check:
            print("[Check] File already exists! loading file")
            df_final = pd.read_csv(name, index_col=0)
            return df_final
        print("[Check] File not found, merging...")
        # Pour chaque catégorie
        for i, cat in enumerate(self.categories):
            # Récupération du DataFrame de la catégorie
            df = self.get_pd_file_from_year(cat=cat, begin=begin, end=end)
            # Si c'est la première catégorie, le DataFrame final est le DataFrame de la catégorie
            if cat == self.categories[0]:
                df["grav"] = df.grav.map({1:0,2:3,3:2,4:1})
                df = pd.DataFrame(df.groupby("Num_Acc")["grav"].max()).reset_index()
                df_final = df
            else:
                # Fusion du DataFrame avec le DataFrame final
                df_final = pd.merge(df_final, df, on="Num_Acc")
        return df_final


    
    def preprocess_df(self, begin=2005, end=2021, second_vehicule=False, check=True, save=True, name="dataset_velo_acc_preprocess.csv"):
        self.second_vehicule = second_vehicule
        path = os.path.join(self.data_dir ,name)
        if os.path.exists(path) and check:
            print("[Check] File already exists! loading file")
            self.df_final = pd.read_csv(path, index_col=0)
            return self.df_final
        elif not os.path.exists(path) and check:
            print("[Preprocessing] File not found, merging...")
        self.df_final = self.get_merge_df(begin,end)
        if self.df_final is not None:
            print("[Preprocessing] File not found, preprocessing...")
            # Sélection des vélos
            Num_Acc =  self.df_final[self.df_final["catv"] == 1]["Num_Acc"]
            df_velo_acc = self.df_final[self.df_final["Num_Acc"].isin(list(Num_Acc))]
            df_velo_acc.reset_index(drop=True, inplace=True)      
            self.df_final = df_velo_acc

            # Ajout du second véhicule
            if self.second_vehicule:
                new_cols = ["manv2", "catv2", "obs2", "obsm2", "choc2"]
                cols = list(df_velo_acc.columns)+new_cols
                df_velo_acc_veh = pd.DataFrame(columns=cols)
                index = -1
                num_acc_inserted: list = []
                df_velo_acc = df_velo_acc.sort_values("Num_Acc")
                for _, row in tqdm(df_velo_acc.iterrows()):
                    if row.Num_Acc in num_acc_inserted:
                        manv2 = row["manv"]
                        catv2 = row["catv"]
                        obs2 = row["obs"]
                        obsm2 = row["obsm"]
                        choc2 = row["choc"]
                        df_velo_acc_veh.loc[index, new_cols] = [manv2, catv2, obs2, obsm2, choc2]
                    else:
                        index += 1
                        df_velo_acc_veh = df_velo_acc_veh.append({col:val for col,val in zip(cols,list(row.to_numpy())+[None]*5)} , ignore_index=True)
                        num_acc_inserted.append(row.Num_Acc)
                self.df_final = df_velo_acc_veh
            
            self._is_prepared = True

            if save:
                self.df_final.to_csv(path)
            return self.df_final
