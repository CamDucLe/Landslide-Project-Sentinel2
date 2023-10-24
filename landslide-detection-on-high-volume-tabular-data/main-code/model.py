import numpy as np
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier

def prepareModel():
    model_name = 'rf.pkl'
    if os.path.exists(model_name):
        print('.......... Loading model ......... \n\n')
        model = pickle.load(open(model_name, 'rb'))
    else:
        print('.......... Creating model ......... \n\n')
        model = RandomForestClassifier(n_estimators=16, max_depth=16, random_state=27, n_jobs=10, class_weight='balanced') # 

    return model
