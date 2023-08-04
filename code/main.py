import tensorflow as tf
import numpy as np 
import os
import platform


from dataset import *
from data_generator import *
from model import *
from train import *
from test import *


def init_argparse():
    parser = argparse.ArgumentParser(
        usage="%(prog)s --is_train XXX --stored_folder XXX --batch_size XXX --is_multi_res XXX",
        description="Settings" 
    )
    
    parser.add_argument("--is_train", required=True, help=' --is_train 1')
    parser.add_argument("--stored_folder", required=True, help=' --stored_folder <output folder>')
    parser.add_argument("--batch_size", required=True, help=' --batch_size 20')
    parser.add_argument("--is_multi_res", required=True, help=' --is_multi_res 1')

    return parser


def main():
    prepareDataset(data_dir='../zip_data') # 
    print('Python version: ',platform.python_version())
    print('Tensorflow version: ',tf.__version__)
    print('Numpy version: ',np.__version__,'\n')

    parser   = init_argparse()
    args     = parser.parse_args()
    
    is_train = int(args.is_train)
    stored_dir      = '../results/' + args.stored_folder + '/'
    best_model_file = stored_dir + 'model.h5'
    BATCH_SIZE      = int(args.batch_size)   # 9 or 15 or 20 or 30 or 36 or 40
    is_multi_resolution = int(args.is_multi_res)

    ## initialize data generator
    generator = DataGenerator(data_dir='../dataset/train', batch_size=BATCH_SIZE, train_ratio=0.8, band_opt=1, is_multi_resolution=is_multi_resolution) # band_opt: 0->14, 1->21, 2->9

    if is_train == 1:
        model = createModel()
        print('### START TRAINING PROCESS ###')
        trainModel(model=model, stored_dir=stored_dir, generator=generator, is_multi_resolution=is_multi_resolution)
    else:
        model = loadModel(best_model_file)  
        if model is not None:
            old_test_f1, old_test_miou = testModel( model=model, generator=generator, 
                                                    best_model_file=best_model_file, stored_dir=stored_dir,
                                                    old_test_f1=0, old_test_miou=0, 
                                                    test_only=True, is_post_processing=0, 
                                                    is_multi_resolution=is_multi_resolution)      
                                                    # is_post_processing=1 : apply thresholding,
                                                    # is_post_processing=2 : apply morphology,  
                               
    return


if __name__ == '__main__':
    main()