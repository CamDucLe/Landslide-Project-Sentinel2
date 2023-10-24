import numpy as np
import pandas as pd


def prepareLabel(df):
    y = df [:,-1]
    y = np.expand_dims(y, axis=1) 
    x = df[:,:-1]
    
    del df
    
    return x, y
