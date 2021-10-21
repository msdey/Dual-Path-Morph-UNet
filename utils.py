import numpy as np
from keras import backend as K
import tensorflow as tf
from keras.losses import binary_crossentropy
from keras.callbacks import Callback


# Loss function
def dice(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice_score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice_score

def dice_loss(y_true, y_pred):
    loss = 1 - dice(y_true, y_pred)
    return loss


def tversky_score(y_true, y_pred):
    smooth = 1
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.5
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky_score(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred):
    pt_1 = tversky_score(y_true, y_pred)
    g = 2
    gamma = 1 / g
    return K.pow((1 - pt_1), gamma)

def bce_focal_tversky_loss(y_true, y_pred):
    return 0.4 * binary_crossentropy(y_true, y_pred) + 0.6 * focal_tversky_loss(y_true, y_pred)

# Metrics

def tp(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    tp = (K.sum(y_pos * y_pred_pos) + smooth) / (K.sum(y_pos) + smooth)
    return tp


def tn(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth)
    return tn

def prec(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    prec = (tp + smooth) / (tp + fp + smooth)
    return prec


def rec(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    recall = (tp + smooth) / (tp + fn + smooth)
    return recall

# Evaluation Metrics
def compute_iou(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)

    if img1.shape[0] != img2.shape[0]:
        raise ValueError("Shape mismatch: the number of images mismatch.")
    IoU = np.zeros((img1.shape[0],), dtype=np.float32)
    for i in range(img1.shape[0]):
        im1 = np.squeeze(img1[i] > 0.5)
        im2 = np.squeeze(img2[i] > 0.5)

        if im1.shape != im2.shape:
            raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
        intersection = np.logical_and(im1, im2)
        if im1.sum() + im2.sum() == 0:
            IoU[i] = 100
        else:
            IoU[i] = 2. * intersection.sum() * 100.0 / (im1.sum() + im2.sum())
    return IoU


def compute_dice(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score
    # Compute F1 score
    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / im_sum


# misc
def normalise(img_arr):
    img_arr = img_arr/255.
    return img_arr

def denormalise(img_arr):
    img_arr = img_arr*255.
    return img_arr.astype(np.uint8)

def binarise(img_arr):
    img_arr = img_arr.point(lambda x: 0 if x<32 else 255, '1')
    return img_arr


