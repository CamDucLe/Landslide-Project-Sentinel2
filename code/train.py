import tensorflow as tf
import random
from datetime import datetime

from losses import *
from metrics import *


def lrScheduler(epoch, lr):
    if epoch < 8:
        return 1e-4 # 1e-4
    elif epoch < 15:
        return 5e-5
    elif epoch < 20:
        return 2e-5
    elif epoch < 25:
        return 1e-5
    elif epoch < 30:
        return 5e-6
    elif epoch < 35:
        return 2e-6
    elif epoch < 42:
        return 1e-6
    elif epoch < 50:
        return 5e-7
    else:
        return 1e-7

def doBackPropagationSingleResolution(model, x_train, y_true_train, epoch_loss, train_acc, AandB_train, AorB_train, intersection_train, union_train):
    with tf.GradientTape() as tape:
        ## predict masks of a batch of images
        y_pred_train = model(x_train) # Bx128x128x2

        # calculte multiple losses on each batch
        loss_1       = model.loss[0](y_true_train, y_pred_train)   # Focal loss
        loss_2       = model.loss[1](y_true_train, y_pred_train)   # IOU loss
        # loss_3       = model.loss[2](y_true_train, y_pred_train)   #
        loss         = 1.0*loss_1 + 1.0*loss_2 # + 3*loss_3      # total loss

        # ## calculte single loss on each batch
        # loss_1 = model.loss[0](y_true_train, y_pred_train)   # loss
        # loss = loss_1              # total loss

        epoch_loss  += loss

        ## do backpropagtion
        grads        = tape.gradient(loss, model.trainable_variables)
        # tf.debugging.check_numerics(grads[0], 'grad contains NaN values.')
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    ## calculte metrics on each batch
    train_acc               += Accuracy(y_true_train, y_pred_train)   # data type: float32
    AandB_t, AorB_t          = F1(y_true_train, y_pred_train)         # data type: float32
    intersection_t, union_t  = MIOU(y_true_train, y_pred_train)       # data type: float32

    AandB_train        += AandB_t
    AorB_train         += AorB_t
    intersection_train += intersection_t
    union_train        += union_t
    
    return model, epoch_loss, train_acc, AandB_train, AorB_train, intersection_train, union_train

def doBackPropagationMultiResolution(model, x_train, y_true_train, epoch_loss, train_acc, AandB_train, AorB_train, intersection_train, union_train):
    with tf.GradientTape() as tape:
        ## predict masks of a batch of images
        y_pred_train = model(x_train) # Bx128x128x2

        # calculte multiple losses on each batch
        loss_1       = model.loss[0](y_true_train[0], y_pred_train[0])   # Focal loss
        loss_2       = model.loss[1](y_true_train[0], y_pred_train[0])   # IOU loss
        loss_256     = 1.0*loss_1 + 1.0*loss_2                     # total loss
        
        loss_1       = model.loss[0](y_true_train[1], y_pred_train[1])   # Focal loss
        loss_2       = model.loss[1](y_true_train[1], y_pred_train[1])   # IOU loss
        loss_128     = 1.0*loss_1 + 1.0*loss_2                     # total loss

        loss_1       = model.loss[0](y_true_train[2], y_pred_train[2])   # Focal loss
        loss_2       = model.loss[1](y_true_train[2], y_pred_train[2])   # IOU loss
        loss_64      = 1.0*loss_1 + 1.0*loss_2                     # total loss
        
        ## do backpropagtion
        epoch_loss  += 0.3*loss_256 + 0.4*loss_128 + 0.3*loss_64
        grads        = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    ## calculte metrics on each batch # TODO other resolution? how to measure ?
    train_acc               += Accuracy(y_true_train[1], y_pred_train[1])   # data type: float32
    AandB_t, AorB_t          = F1(y_true_train[1], y_pred_train[1])         # data type: float32
    intersection_t, union_t  = MIOU(y_true_train[1], y_pred_train[1])       # data type: float32

    AandB_train        += AandB_t
    AorB_train         += AorB_t
    intersection_train += intersection_t
    union_train        += union_t
    
    return model, epoch_loss, train_acc, AandB_train, AorB_train, intersection_train, union_train


def trainModel(model, stored_dir, generator, is_multi_resolution):
    train_dir       = '../dataset/train/'
    # val_dir         = '../dataset/val/'
    # test_dir        = '../dataset/test/'

    best_model_file = stored_dir + 'model.h5'

    current_epoch   = 0
    old_test_miou   = 0
    old_test_f1     = 0

    num_batch_train, last_batch_train = generator.getNumBatch(op='train')
    num_batch_test, last_batch_test   = generator.getNumBatch(op='val')

    print("batch train: ", num_batch_train, last_batch_train)
    print("batch test: ",num_batch_test, last_batch_test)

    n_train_imgs  = generator.getNumImg(op='train')
    n_test_imgs   = generator.getNumImg(op='val')


    for epoch in range(current_epoch, 65):
        print('\n\n ==================== Epoch ', epoch,'======================')
        start_train = datetime.now()
        train_acc    = 0
        train_f1     = 0
        train_miou   = 0
        epoch_loss   = 0

        AandB_train        = tf.zeros_like([0,0], dtype=tf.float32)
        AorB_train         = tf.zeros_like([0,0], dtype=tf.float32)
        intersection_train = tf.zeros_like([0,0], dtype=tf.float32)
        union_train        = tf.zeros_like([0,0], dtype=tf.float32)

        ## learning rate scheduling

        model.optimizer.learning_rate = lrScheduler(epoch, model.optimizer.learning_rate.numpy())
        print(' *** learning rate: ', model.optimizer.learning_rate)

        print('-------- training ---------')
        for batch_train_idx in range(num_batch_train):
            ## avoid dead state + test
            if batch_train_idx == int(num_batch_train/2):
                print("........  50% ..........")
                if epoch > 40:
                    old_test_f1, old_test_miou = testModel( model=model, generator=generator,
                                                            best_model_file=best_model_file, stored_dir=stored_dir,
                                                            old_test_f1=old_test_f1, old_test_miou=old_test_miou,
                                                            is_multi_resolution=is_multi_resolution)
            if batch_train_idx == (num_batch_train-1):
                print("........  100% ..........")
     
            ## get x_train, y_train from data generator
            if is_multi_resolution:
                if epoch > 60:
                    x_train, y_true_train_256, y_true_train_128, y_true_train_64, n_imgs = generator.getBatch(batch_num=batch_train_idx, is_aug=False, is_train=True, is_cutmix=False)
                else:
                    x_train, y_true_train_256, y_true_train_128, y_true_train_64, n_imgs = generator.getBatch(batch_num=batch_train_idx, is_aug=True, is_train=True, is_cutmix=True)
                y_true_train = [y_true_train_256, y_true_train_128, y_true_train_64]
                ## optimization process
                model, epoch_loss, train_acc, AandB_train, AorB_train, intersection_train, union_train = doBackPropagationSingleResolution(model, x_train, y_true_train, epoch_loss, train_acc,
                                                                                                                       AandB_train, AorB_train, intersection_train, union_train)
            
            else:
                if epoch > 60:
                    x_train, y_true_train , n_imgs = generator.getBatch(batch_num=batch_train_idx, is_aug=False, is_train=True, is_cutmix=False)
                else:
                    x_train, y_true_train , n_imgs = generator.getBatch(batch_num=batch_train_idx, is_aug=True, is_train=True, is_cutmix=True)
                y_true_train = y_true_train

                ## optimization process
                model, epoch_loss, train_acc, AandB_train, AorB_train, intersection_train, union_train = doBackPropagationSingleResolution(model, x_train, y_true_train, epoch_loss, train_acc,
                                                                                                                       AandB_train, AorB_train, intersection_train, union_train)
            
            # with tf.GradientTape() as tape:
            #     ## predict masks of a batch of images
            #     y_pred_train = model(x_train) # Bx128x128x2

            #     # calculte multiple losses on each batch
            #     loss_1       = model.loss[0](y_true_train, y_pred_train)   # Focal loss
            #     loss_2       = model.loss[1](y_true_train, y_pred_train)   # IOU loss
            #     # loss_3       = model.loss[2](y_true_train, y_pred_train)   #
            #     loss         = 1.0*loss_1 + 1.0*loss_2 # + 3*loss_3      # total loss

            #     # ## calculte single loss on each batch
            #     # loss_1 = model.loss[0](y_true_train, y_pred_train)   # loss
            #     # loss = loss_1              # total loss

            #     epoch_loss  += loss

            #     ## do backpropagtion
            #     grads        = tape.gradient(loss, model.trainable_variables)
            #     # tf.debugging.check_numerics(grads[0], 'grad contains NaN values.')
            #     model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # ## calculte metrics on each batch
            # train_acc               += Accuracy(y_true_train, y_pred_train)   # data type: float32
            # AandB_t, AorB_t          = F1(y_true_train, y_pred_train)         # data type: float32
            # intersection_t, union_t  = MIOU(y_true_train, y_pred_train)       # data type: float32

            # AandB_train        += AandB_t
            # AorB_train         += AorB_t
            # intersection_train += intersection_t
            # union_train        += union_t

        ## calculte metrics on each train epoch
        if is_multi_resolution:
            TODO = True
        else:
            train_acc /= n_train_imgs*128*128
            train_f1   = calculateF1(AandB_train, AorB_train)
            train_miou = calculateMIOU(intersection_train, union_train)

        ## testing ...
        if epoch >= 1:
            old_test_f1, old_test_miou = testModel( model=model, generator=generator,
                                                    best_model_file=best_model_file, stored_dir=stored_dir,
                                                    old_test_f1=old_test_f1, old_test_miou=old_test_miou, 
                                                    is_multi_resolution=is_multi_resolution)

        ## write log
        with open(os.path.join(stored_dir,"train_log.txt"), "a") as text_file:
            text_file.write("Epoch: {}; lr: {}; Train miou: {}; Train f1: {}; Train accuracy: {}\n".format(epoch, model.optimizer.learning_rate.numpy(), train_miou, train_f1 ,train_acc))

        print('===>>> Train loss:  ', epoch_loss.numpy(), '; Train MeanIOU: ', train_miou, '; Train F1: ', train_f1, '; Train Accuracy: ', train_acc)
        print('===>>> epoch training time: ', datetime.now()-start_train, '\n\n')


    return
