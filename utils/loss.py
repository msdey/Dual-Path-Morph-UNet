from keras import backend as K
from keras.losses import binary_crossentropy


def dice(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice_score = (2.0 * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    )
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
    return (true_pos + smooth) / (
        true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth
    )


def tversky_loss(y_true, y_pred):
    return 1 - tversky_score(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred):
    pt_1 = tversky_score(y_true, y_pred)
    g = 2
    gamma = 1 / g
    return K.pow((1 - pt_1), gamma)


def bce_focal_tversky_loss(y_true, y_pred):
    return 0.4 * binary_crossentropy(y_true, y_pred) + 0.6 * focal_tversky_loss(
        y_true, y_pred
    )
