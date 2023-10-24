import os
import tensorflow as tf
import numpy as np

from losses import *

### list your models here ###
#from SRAU_Net import *
from RAU_Net import *



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



## for file testing purpose
if __name__ == '__main__':
    if os.path.exists('../raesults/23-mono-f-i-23/model.h5'):
        model = loadModel()
    else:
        model = createModel()
