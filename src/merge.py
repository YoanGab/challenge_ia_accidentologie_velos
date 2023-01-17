
from utils import *

import pandas as pd
import numpy as np

from geopy.distance import great_circle


def filter_top_nine_citys_only(df_acc_, df_infra_):
    top_nine_citys = [75,44,33,59,31,13,69,34,67]
    mask_infra = np.zeros(df_infra_.shape[0]).astype(bool)
    mask_acc = np.zeros(df_acc_.shape[0]).astype(bool)

    for code in top_nine_citys:
        mask_acc = np.logical_or(mask_acc, df_acc_['com'] == code)
        mask_infra = np.logical_or(mask_infra, np.logical_or(df_infra_['code_com_g'] == code, df_infra_['code_com_d'] == code))

    df_infra_ = df_infra_[mask_infra]
    df_acc_ = df_acc_[mask_acc]

    return df_acc_.reset_index(drop=True), df_infra_.reset_index(drop=True)

#usage : df_acc, df_infra = filter_top_nine_citys_only(df_acc, df_infra)



def associate_infra_index_to_acc(df_acc, df_infra):
    """ 
    Associate for each accident the index of an infrastructure
    """
    df_acc_ = df_acc.iloc[:64, :].copy()
    df_infra_ = df_infra.copy()

    top_nine_citys = [75,44,33,59,31,13,69,34,67]

    df_acc_["distance_to_infra"] = np.nan
    df_acc_["infra_index"] = np.nan

    for code in top_nine_citys:

        print(code)

        #Select a sub-dataset (a city) in order to reduce research computations among infrastructure
        mask_acc = df_acc_['com'] == code
        mask_infra = np.logical_or(df_infra_['code_com_g'] == code, df_infra_['code_com_d'] == code)
        df_infra_city = df_infra_[mask_infra]
        df_acc_city = df_acc_[mask_acc]
        
        for i_acc in range(df_acc_city.shape[0]):
            acc = df_acc_city.iloc[i_acc,:]
            acc_index = df_acc_city.iloc[i_acc:i_acc+1,:].index.values[0] #subtefuge pour obtenir l'index
            lat_acc = acc['lat']
            long_acc = acc['long']
            
            #calcul la distance d'un accident à tout les infrastructure d'une même ville
            dist_series = df_infra_city[['latitude_dep', 'longitude_dep', 'latitude_fin', 'longitude_fin']].apply(distance_infra_acc, axis=1, args=(lat_acc, long_acc))
            
            #Selectionne l'infra la plus proche
            df_acc_.loc[acc_index, "infra_index"] = dist_series.argmin() #reference index of infrastructure
            df_acc_.loc[acc_index, "distance_to_infra"] = dist_series.min()
    
    return df_acc_, df_infra_
    #Usage  : df_acc, df_infra = associate_infra_index_to_acc(df_acc, df_infra)


def merge(df_acc, df_infra):

    df_acc, df_infra = filter_top_nine_citys_only(df_acc, df_infra)
    df_acc, df_infra = associate_infra_index_to_acc(df_acc, df_infra)

    return df_acc, df_infra