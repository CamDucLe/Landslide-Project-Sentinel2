import numpy as np
import pandas as pd
import pyarrow as pa
import psutil
import random
import sklearn
from sklearn.cluster import DBSCAN
from datetime import datetime
from collections import Counter


def dfscanWholeDataset():
    pos_file_name = '/home/phaml/storage/01_project/04_AI_for_Earth/05_landslide_dataset/chunks/pos/carinthia_slides.arrow'
    neg_dir = '/home/phaml/storage/01_project/04_AI_for_Earth/05_landslide_dataset/chunks/neg/'

    ## read .arrow file of positive samples
    pos_df = pa.ipc.open_file(pos_file_name).read_all()
    pos_df = pos_df.to_pandas()
    pos_df = pos_df.dropna(axis=0)

    ## read .arrow file of negative samples
    chunks  = []
    chunks.append(pos_df)
    for i in range(1,10):
        neg_file_name = neg_dir + 'partition=' + str(i) + '/' + 'part-0.arrow'
        neg_df = pa.ipc.open_file(neg_file_name).read_all()
        neg_df = neg_df.to_pandas()
        neg_df = neg_df.dropna(axis=0)
        chunks.append(neg_df)

    df = pd.concat(neg_chunks)
    del neg_chunks, pos_df, neg_df

    dbscan = DBSCAN(eps=TODO, min_samples=TODO, metric='euclidean', n_jobs=8)
    dbscan = dbscan.fit(df)
        
    print(Counter(dbscan.labels_))


    return 

