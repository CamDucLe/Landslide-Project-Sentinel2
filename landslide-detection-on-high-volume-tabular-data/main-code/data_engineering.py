import pandas as pd
import numpy as np
import random
import pickle

from datetime import datetime
#from load_data import *
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder,LabelEncoder,StandardScaler



def encodeCategoricalFeature(df, is_train=True):
    for col in df.columns:
        if df[col].dtype in ['category', 'bool']:
            if is_train:
                ## initialize and fit + transform train data
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform( df[col] )
                ## save encoder
                encoder_name = 'pklFolder/' + col + '_encoder.pkl'  
                pickle.dump(encoder, open(encoder_name, 'wb'))
            else:
                ## load encoder
                encoder_name = 'pklFolder/' + col + '_encoder.pkl'
                encoder = pickle.load(open(encoder_name, 'rb'))
                ## transform test data 
                df[col] = encoder.transform(df[col])

    return df


def scaleFeatures(df, op='MinMax', is_train=True):
    tmp_df = df.drop(columns=['x', 'y', 'slide'])
    if is_train:
        ## initialize and fit + transform train data
        scaler = MinMaxScaler()
        tmp_df = scaler.fit_transform(tmp_df)
        ## save encoder
        pickle.dump(scaler, open('pklFolder/MinMaxScaler.pkl', 'wb'))
        ## construct numpy array as X
        df = np.concatenate((tmp_df, df[['slide']].to_numpy() ), axis=1)
    else:
        ## load encoder + transform test data
        scaler = pickle.load(open('pklFolder/MinMaxScaler.pkl', 'rb'))
        tmp_df = scaler.transform(tmp_df)
        ## construct numpy array as X
        df = np.concatenate((tmp_df, df[['slide']].to_numpy() ), axis=1)

    del tmp_df

    return df


def doFeatureEngineering(df, is_train=True):
    df = encodeCategoricalFeature(df=df, is_train=is_train) # pandas data frame
    df = scaleFeatures(df=df, is_train=is_train)            # numpy data frame
    
    return df

