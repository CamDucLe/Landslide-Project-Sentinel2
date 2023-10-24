import numpy as np 
import pandas as pd
import pickle

import sklearn
from sklearn import metrics
from load_data import *
from data_engineering import *
from label_prepare import *
from model import *
from datetime import datetime
from argparse import ArgumentParser


def initArgparse():
    parser = ArgumentParser(
        usage="%(prog)s --is_train XXX",
        description="Settings"
    )

    parser.add_argument("--is_train", required=True, help=' --is_train 1')

    return parser


def main(is_train=True):
    ## parser vs arguments ##
    parser   = initArgparse()
    args     = parser.parse_args()

    is_train = bool(int(args.is_train))
    if is_train:
        print('.......... Training .......... \n\n')
    else:
        print('.......... Testing .......... \n\n')


    ## load data ##
    df_train = loadChunks(is_train=True)  # pandas data_frame
    df_test  = loadChunks(is_train=False) # pandas data_frame
    print('\n\n=========== Load data ==========')
    print(df_train.shape, df_test.shape)

    ## pre-process data ##
    df_train = doFeatureEngineering(df_train, is_train=True)    # NxF numpy array (last column is label column)
    df_test  = doFeatureEngineering(df_test,  is_train=False)   # NxF numpy array (last column is label column)
    print('\n\n=========== Pre-process data ==========')
    print(df_train.shape, df_test.shape)

    ## label prepare ## 
    x_train, y_train = prepareLabel(df_train)  # Nx(F-1) and Nx1 numpy array
    x_test, y_test   = prepareLabel(df_test)   # Nx(F-1) and Nx1 numpy array
    print('\n\n=========== Prepare label  ==========')
    print('X-train and Y-train shape: ', x_train.shape, y_train.shape)
    print('X-test and Y-test shape: ', x_test.shape, y_test.shape)

    ## load or create model ##
    model = prepareModel()
    print('\n\n=========== Prepare model === =======')
    print(model)
    
    #del model
    #exit()
    #return

    ## train or test ## 
    if is_train:
        print('\n\n=========== Train ==========')
        start_train = datetime.now()
        model.fit(x_train, y_train)
        print('Training time: ', datetime.now()-start_train )
        print('Train Score:', metrics.classification_report(y_train, model.predict(x_train)))
        ## save the model  ##
        pickle.dump(model, open('rf.pkl', 'wb'))
    else:
        print('\n\n=========== Test  =======')
        y_pred = model.predict(x_test)
        result = metrics.classification_report(y_test, y_pred)
        print('Final Result: ', result)

    return 

if __name__ == "__main__":
    main()
    
