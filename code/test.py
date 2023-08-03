import tensorflow as tf
import random
from datetime import datetime
from metrics import *


def testModel(model, generator, best_model_file, stored_dir,  old_test_f1, old_test_miou, test_only=False, is_post_processing=0):
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
        x_test, y_true_test , n_imgs = generator.getBatch(batch_num=batch_test_idx, is_aug=False, is_train=False, is_cutmix=False)
        y_pred_test         = model(x_test)
        if is_post_processing == 1:
            y_pred_test = postProcessingPixelLevelThresholding(y_pred_test, threshold=0.72) # apply thresholding
        elif is_post_processing == 2:
            y_pred_test = postProcessingMorphology(y_pred_test, op=1) # apply thresholding (remove pepper noise)
        elif is_post_processing == 3:
            y_pred_test = postProcessingAll(y_pred_test)  # apply all post_processing techniques

        ## calculte metrics on each batch
        test_acc                += Accuracy(y_true_test, y_pred_test)   # data type: float32
        AandB_t, AorB_t          = F1(y_true_test, y_pred_test)         # data type: float32
        intersection_t, union_t  = MIOU(y_true_test, y_pred_test)       # data type: float32

        AandB_test        += AandB_t
        AorB_test         += AorB_t
        intersection_test += intersection_t
        union_test        += union_t

    ## calculte metrics on each test epoch 
    test_acc /= n_test_imgs*128*128
    test_f1   = calculateF1(AandB_test, AorB_test)
    test_miou = calculateMIOU(intersection_test, union_test)
    print('>>>>> Test MeanIOU: ', test_miou, '; Test F1: ', test_f1, '; Test Accuracy: ', test_acc, '  and Testing Time : ', datetime.now()-start_test)

    ## Save model when successfully testing
    if (test_f1 > old_test_f1) or (test_miou > old_test_miou):
        old_test_f1   = test_f1
        old_test_miou = test_miou
        if test_only == False:
            model.save(best_model_file)
            print('...... Save model completed ......' )
            with open(os.path.join(stored_dir,"train_log.txt"), "a") as text_file:
                text_file.write("\n --- Save best model at Epoch: {}; MIOU: {}; F1: {} ---\n\n".format(epoch, old_test_miou, old_test_f1))

    return old_test_f1, old_test_miou