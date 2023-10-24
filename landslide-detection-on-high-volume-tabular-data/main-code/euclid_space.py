import numpy as np
import pandas as pd
import pyarrow as pa
import psutil
import random
from datetime import datetime



def calculateDistance():
    pos_file_name = '/home/phaml/storage/01_project/04_AI_for_Earth/05_landslide_dataset/chunks/pos/carinthia_slides.arrow'
    neg_dir = '/home/phaml/storage/01_project/04_AI_for_Earth/05_landslide_dataset/chunks/neg/'
    
    ## read .arrow file of positive samples
    pos_df = pa.ipc.open_file(pos_file_name).read_all()
    pos_df = pos_df.to_pandas()
    pos_df = pos_df.dropna(axis=0)
    pos_df = pos_df.drop(columns=['lithology', 'land_cover', 'geomorphons', 'slide', 'x', 'y']) # get rid of categorical, x, y and label

    ## read .arrow file of negative samples
    neg_chunks  = []
    for i in range(1,10):
        neg_file_name = neg_dir + 'partition=' + str(i) + '/' + 'part-0.arrow'
        neg_df = pa.ipc.open_file(neg_file_name).read_all()
        neg_chunks.append(neg_df.to_pandas())
    
    neg_df = pd.concat(neg_chunks)
    del neg_chunks
    neg_df = neg_df.dropna(axis=0)
    neg_df = neg_df.drop(columns=['lithology', 'land_cover', 'geomorphons', 'slide', 'x', 'y']) # get rid of categorical, x, y and label
    
    ## find center of positive and negtive samples
    pos_center = np.array(pos_df.sum()/pos_df.shape[0])
    neg_center = np.array(neg_df.sum()/neg_df.shape[0])
    print('\n\n Pos Center: ', pos_center)
    print('\n\n Neg Center: ', neg_center)
    print('\n\n Euclid distance between them: ', np.sqrt(np.sum(np.power(pos_center-neg_center,2))))

    ## find farest point 
    max_pos_dis = np.max(np.sqrt(np.sum(np.power(pos_df.to_numpy() - pos_center,2), axis=1)))
    max_neg_dis = np.max(np.sqrt(np.sum(np.power(neg_df.to_numpy() - neg_center,2), axis=1)))
    print('\n\n Max pos distance: ', max_pos_dis)
    print('\n\n Max neg distance: ', max_neg_dis)

    return 


if __name__ == "__main__":
    calculateDistance()

