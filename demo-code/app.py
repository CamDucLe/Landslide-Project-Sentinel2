import gradio as gr
import tensorflow as tf
import numpy as np
import os

from PIL import Image

from data_generator import *
from model import *
from test import *
from util import *



def doSegmentation(batch_num):
    ## initialize data generator
    generator = DataGenerator(data_dir='data', batch_size=1, train_ratio=1., band_opt=1, is_multi_resolution=1) # band_opt: 0->14, 1->23, 2->9
    
    ## load model
    model = loadModel("model.h5")
    
    ## run Data_Generator
    x_test, y_true_test_256, y_true_test_128, y_true_test_64, n_imgs = generator.getBatch(batch_num=int(batch_num), is_aug=False, is_train=True, is_cutmix=False)
    
    ## make predictions
    y_pred_test = model(x_test)
    
    ##  convert image/masks to PIL img
    img = x_test[0]                   # 128x128x23
    img = changeBGR2RGB(img[:,:,1:4]) # 128x128x3
    # print(img.shape, np.max(img), np.min(img))
    img = Image.fromarray(np.uint8(img*255))
    
    true_mask = y_true_test_128[0] # 128x128
    true_mask = true_mask.numpy()  # 128x128
    # print(true_mask.shape, np.max(true_mask), np.min(true_mask))
    true_mask = Image.fromarray(np.uint8(true_mask*255))

    pred_mask_1 = tf.math.argmax(y_pred_test[1], axis=-1)[0] # 128x128
    pred_mask_1 = pred_mask_1.numpy()                         # 128x128
    # print(pred_mask.shape, np.max(pred_mask), np.min(pred_mask))
    pred_mask_1 = Image.fromarray(np.uint8(pred_mask_1*255))

    ## do post processing
    # y_pred_test[1] = doPostProcessing(x_test, y_pred_test, model, 1)
    y_pred_test[1] = postProcessingPixelLevelThresholding(y_pred_test[1], threshold=0.95)
    pred_mask_2 = tf.math.argmax(y_pred_test[1], axis=-1)[0]  # 128x128
    pred_mask_2 = pred_mask_2.numpy()                         # 128x128
    pred_mask_2 = Image.fromarray(np.uint8(pred_mask_2*255))

    
    ## init list of return images/masks
    return_imgs = [img, true_mask, pred_mask_1, pred_mask_2]
    # return img, true_mask, pred_mask_1, pred_mask_2
    return return_imgs
    

with gr.Blocks(title="Landslide-Segmentation-Demo") as intf:
    with gr.Column(variant="panel"):
        with gr.Row():
            slider = gr.Slider(0, 20, step=1, label="Change slider to see different landslide images, masks and predictions (only 21 examples ^^)")
            btn = gr.Button("Do Segmenting", scale=0)
        with gr.Row():
            # text = gr.Button(value="Remote Sensing Image   ==>>   True_mask (binary mask)   ==>>   Pred_mask (model's prediction only)   ==>>   Pred_mask applied post-processing (thresholding)", interactive=False)
            text = gr.Button(value="Remote Sensing Image", interactive=False)
            text = gr.Button(value="True_mask (binary mask)", interactive=False)
            text = gr.Button(value="Pred_mask (model's prediction only)", interactive=False)
            text = gr.Button(value="Pred_mask applied post-processing (thresholding)", interactive=False)
        gallery = gr.Gallery( columns=4 )

    btn.click(doSegmentation, slider, gallery)    

intf.launch() 
