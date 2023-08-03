import tensorflow as tf
import numpy as np
import cv2


def postProcessingPixelLevelThresholding(masks_pred, threshold=0.5):
  """
    # apply thresholding
    # input:
        - masks_pred  (Bx128x128x2):
        - threshold  (default=0.5): smaller threshold -> more focus on landslide pixel, larger threshold -> strictly focus on landslide pixel,
    # output:
        - mask_list  (Bx128x128x2):
  """
  masks_pred = masks_pred.numpy()
  c1 = masks_pred[..., 0]
  c2 = np.where(masks_pred[..., 1] > threshold, 1, 0)
  c1 = np.expand_dims(c1,axis=-1)
  c2 = np.expand_dims(c2,axis=-1)
  new_masks = np.concatenate((c1,c2), axis=-1)
  new_masks = tf.convert_to_tensor(new_masks, dtype=tf.float32)

  return new_masks


def postProcessingMorphology(masks_pred, op=1):
  """
    # apply thresholding
    # input:
        - masks_pred  (Bx128x128x2):
    # output:
        - mask_list  (Bx128x128x2):
  """
  masks_pred = masks_pred.numpy()
  masks_pred = np.argmax(masks_pred, axis=-1) # Bx128x128
  kernel = np.ones((4, 4), np.uint8)
  if op == 0: # opening - removing salt noise 
    masks_erosion = cv2.erode(masks_pred, kernel, iterations=1)
    new_masks     = cv2.dilate(masks_erosion, kernel, iterations=1)
  elif op == 1:       # closing - removing pepper noise
    mask_dilation = cv2.dilate(masks_pred, kernel, iterations=1)
    new_masks     = cv2.erode(masks_dilation, kernel, iterations=1)
  else:
    new_masks     = masks_pred
  
  new_masks = tf.convert_to_tensor(new_masks, dtype=tf.int8)
  new_masks = tf.one_hot(new_masks, depth=2, axis=-1)

  return tf.cast(new_masks, dtype=tf.float32)

def postProcessingAll(masks_pred):
  masks_pred = postProcessingPixelLevelThresholding(masks_pred, threshold=0.72)
  masks_pred = postProcessingMorphology(masks_pred, op=1)

  return masks_pred