import tensorflow as tf
import random
import os

from datetime import datetime
from metrics import *
from post_processing import *



def calculteMetricTest(y_true_test, y_pred_test, test_acc, AandB_test, AorB_test, intersection_test, union_test):
    test_acc                += Accuracy(y_true_test, y_pred_test)   # data type: float32
    AandB_t, AorB_t          = F1(y_true_test, y_pred_test)         # data type: float32
    intersection_t, union_t  = MIOU(y_true_test, y_pred_test)       # data type: float32

    AandB_test        += AandB_t
    AorB_test         += AorB_t
    intersection_test += intersection_t
    union_test        += union_t

    return test_acc, AandB_test, AorB_test, intersection_test, union_test
 

def doPostProcessing(x_test, y_pred_test, model, is_post_processing=0):
    ## apply thresholding
    if is_post_processing == 1:
        y_pred_test_1 = postProcessingPixelLevelThresholding(y_pred_test[1], threshold=0.95) 
    
    ## use morphology operation (remove pepper/salt noise)
    elif is_post_processing == 2:
        y_pred_test_1 = postProcessingMorphology(y_pred_test[1], op=1) 
    
    ## voting between 3 resolution output masks
    elif is_post_processing == 3:
        y_pred_test_1 = postProcessingVotingMask(y_pred_test) 
    
    ## multi angle predictions
    elif is_post_processing == 4 or is_post_processing == 5:
        y_pred_test_90  = model(tf.image.rot90(x_test, k=1))
        y_pred_test_180 = model(tf.image.rot90(x_test, k=2))
        y_pred_test_270 = model(tf.image.rot90(x_test, k=3))
        
        #if is_post_processing == 5:
        #    y_pred_test_90[1] = postProcessingPixelLevelThresholding(y_pred_test_90[1], threshold=0.95) # apply thresholding
        #    y_pred_test_180[1] = postProcessingPixelLevelThresholding(y_pred_test_180[1], threshold=0.95) # apply thresholding
        #    y_pred_test_270[1] = postProcessingPixelLevelThresholding(y_pred_test_270[1], threshold=0.95) # apply thresholding
        y_pred_test_1  = postProcessingMultiAngle(y_pred_test[1],y_pred_test_90[1],y_pred_test_180[1],y_pred_test_270[1]) 

        ## combine multi-angle and thresholding
        if is_post_processing == 5:
            y_pred_test_1 = postProcessingPixelLevelThresholding(y_pred_test_1, threshold=0.95) # apply thresholding
    else:
        y_pred_test_1 =  y_pred_test[1]

    return y_pred_test_1


def testModel(model, generator, best_model_file, stored_dir,  old_test_f1, old_test_miou, test_only=False, is_post_processing=0, is_multi_resolution=0):
    print("------ testing -------")
    num_batch_test, last_batch_test  = generator.getNumBatch(op='val')
    n_test_imgs   = generator.getNumImg(op='val')

    start_test  = datetime.now()
    test_acc    = 0
    test_f1     = 0
    test_miou   = 0

    AandB_test        = tf.zeros_like([0,0], dtype=tf.float32)
    AorB_test         = tf.zeros_like([0,0], dtype=tf.float32)
    intersection_test = tf.zeros_like([0,0], dtype=tf.float32)
    union_test        = tf.zeros_like([0,0], dtype=tf.float32)

    for batch_test_idx in range(num_batch_test):
        ## multi-resolution segmentation head
        if is_multi_resolution:
            x_test, y_true_test_256, y_true_test_128, y_true_test_64, n_imgs = generator.getBatch(batch_num=batch_test_idx, is_aug=False, is_train=False, is_cutmix=False)
            
            ## make predictions
            #start_pred  = datetime.now()
            y_pred_test = model(x_test)
            #print('Inference time of XX WxHxC images using 1 gpu <gpu_name>: ', datetime.now()-start_pred)
            #exit()
            #return

            ## do post processing
            y_pred_test[1] = doPostProcessing(x_test, y_pred_test, model, is_post_processing)
            
            ## calculte metrics on each batch
            test_acc, AandB_test, AorB_test, intersection_test, union_test = calculteMetricTest(y_true_test_128, y_pred_test[1], test_acc, AandB_test, AorB_test, intersection_test, union_test)

        ## single resolution
        else:
            x_test, y_true_test , n_imgs = generator.getBatch(batch_num=batch_test_idx, is_aug=False, is_train=False, is_cutmix=False)
            y_pred_test = model(x_test)
            y_pred_test = doPostProcessing(y_pred_test, is_post_processing)

            ## calculte metrics on each batch
            test_acc, AandB_test, AorB_test, intersection_test, union_test = calculteMetricTest(y_true_test, y_pred_test, test_acc, AandB_test, AorB_test, intersection_test, union_test)


    ## calculte metrics on each test epoch 
    test_acc /= n_test_imgs*128*128
    test_f1   = calculateF1(AandB_test, AorB_test)
    test_miou = calculateMIOU(intersection_test, union_test)
    
    ## write log after testing finished (in training phase only)
    if test_only == False:
        with open(os.path.join(stored_dir,"train_log.txt"), "a") as text_file:
            text_file.write(">>>>> Test Scores >>> test_miou: {}; test_f1: {}; test Acc: {}; test_time: {} ---\n\n".format(test_miou, test_f1, test_acc, datetime.now()-start_test))
    
    ## print test results
    print('>>>>> Test MeanIOU: ', test_miou, '; Test F1: ', test_f1, '; Test Accuracy: ', test_acc, '  and Testing Time : ', datetime.now()-start_test)

    ## Save model when successfully testing
    if (test_f1 > old_test_f1) or (test_miou > old_test_miou):
        old_test_f1   = test_f1
        old_test_miou = test_miou
        if test_only == False:
            model.save(best_model_file)
            print('...... Save model completed ......' )
            with open(os.path.join(stored_dir,"train_log.txt"), "a") as text_file:
                text_file.write("------- Save best model ! TEST_MIOU: {}; TEST_F1: {} -------\n\n".format(old_test_miou, old_test_f1))
        else:
            print('Test F1: ', test_f1)
            print('Test MIOU: ', test_miou)

    return old_test_f1, old_test_miou
