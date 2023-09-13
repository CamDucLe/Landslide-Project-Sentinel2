import os
import tensorflow as tf
import numpy as np

#from U_Net import *
#from U_Net_multi import *
#from W_Net import *
#from S_Net import *
#from M_Net import *
#from Small_CU_Net import *
#from CU_Net import *
#from D_Net import *
#from R_Net import *
#from SR_Net import *
from SRA_Net import *
#from RA_Net import *
from losses import *

def loadModel(model_path):
    if not os.path.exists(model_path):
        print('NOT EXIST ANY MODEL !!!')
        return None

    print('........ loading model ! ........')
    model = tf.keras.models.load_model( filepath=model_path,
                                      custom_objects={ 'FocalLoss': FocalLoss, 'IOULoss': IOULoss})
    print(model.loss)
    print(model.optimizer.learning_rate)
    # model.summary()
    
    return model

def createModel():
    print('........ creating model ! .........')
    model = getModel()
    print(model.loss)
    print(model.optimizer.learning_rate)
    model.summary()
    
    return model

if __name__ == '__main__':
    if os.path.exists('../results/20-mono-f-i-21/model.h5'):
        model = loadModel()
    else:
        model = createModel()