import numpy as np
import numba
from numba import njit, prange
import matplotlib.pyplot as plt
from shapely.geometry import LineString


def precision_recall_at_threshold(y_true, y_pred, threshold):
    smooth = 1
    y_pred_pos = (y_pred >= threshold).astype(bool)
    y_pos = (y_true >= 0.5).astype(bool)
    y_pred_neg = 1 - y_pred_pos
    y_neg = 1 - y_pos
    tp = np.sum(y_pos * y_pred_pos)
    tn = np.sum(y_neg * y_pred_neg)
    fp = np.sum(y_neg * y_pred_pos)
    fn = np.sum(y_pos * y_pred_neg)
    precision = (tp + smooth) / (tp + fp + smooth)  # precision
    recall = (tp + smooth) / (tp + fn + smooth)  # recall
    # tnr = (true_negative + smooth) / (true_negative + fp + smooth)
    return precision, recall


@numba.jit(parallel=True)
def relaxed_precision_recall_at_threshold(y_true, y_pred, threshold):
    y_true = (y_true >= 0.5).astype("float32")
    y_pred = (y_pred >= threshold).astype("float32")

    n_true = len(np.where(y_true == 1.0)[0])
    n_pred = len(np.where(y_pred == 1.0)[0])

    precision_count = 0
    recall_count = 0
    height, width = y_true.shape
    rho = 3  # radius for calculating relaxed metrics

    for row in prange(height):
        for col in prange(width):
            y_true_val = y_true[row][col]
            y_pred_val = y_pred[row][col]

            # get window limits
            rowmin = max(0, row - rho)
            rowmax = min(height, row + rho)
            colmin = max(0, col - rho)
            colmax = min(width, col + rho)

            y_true_win = y_true[rowmin:rowmax, colmin:colmax]
            y_pred_win = y_pred[rowmin:rowmax, colmin:colmax]

            if y_pred_val == 1:
                if np.max(y_true_win) > 0:
                    precision_count += 1

            if y_true_val == 1:
                if np.max(y_pred_win) > 0:
                    recall_count += 1

    # get fractions
    if n_true == 0:
        relaxed_recall = 0
    else:
        relaxed_recall = 1.0 * recall_count / n_true

    if n_pred == 0:
        relaxed_precision = 0
    else:
        relaxed_precision = 1.0 * precision_count / n_pred

    if (relaxed_recall > 0) and (relaxed_precision > 0):
        relaxed_f1 = (
            2
            * relaxed_precision
            * relaxed_recall
            / (relaxed_precision + relaxed_recall)
        )
    else:
        relaxed_f1 = 0
    return relaxed_precision, relaxed_recall


def calculate_breakeven_point(prec, recall):
    breakeven = []
    x = np.linspace(0, 1, 100)
    f = x
    plt.plot(x, f)
    plt.plot(prec, recall)
    first_line = LineString(np.column_stack((x, f)))
    second_line = LineString(np.column_stack((prec, recall)))
    intersection = first_line.intersection(second_line)

    if intersection.geom_type == "MultiPoint":
        plt.plot(*LineString(intersection).xy, "o")
        s, t = LineString(intersection).xy
        breakeven.append(t[1])
    elif intersection.geom_type == "Point":
        plt.plot(*intersection.xy, "o")
        s, t = intersection.xy
        breakeven.append(t[0])
    plt.close()
    return breakeven[0]


def calculate_precision_recall(y_true, y_pred, threshold, relaxed=False):
    precision = []
    recall = []
    precision.append(0)
    recall.append(1)

    for th in threshold:
        precision_th = []
        recall_th = []
        for k in range(len(y_true)):
            if relaxed == False:
                prec, rec = precision_recall_at_threshold(
                    y_true[k, :, :], y_pred[k, :, :], th
                )
            else:
                prec, rec = relaxed_precision_recall_at_threshold(
                    y_true[k, :, :], y_pred[k, :, :], th
                )

            precision_th.append(prec)
            recall_th.append(rec)
        precision.append(np.mean(np.array(precision_th)))
        recall.append(np.mean(np.array(recall_th)))
    return precision, recall


def iou_f1_score(y_true, y_pred):
    smooth = 1
    y_pred_pos = (y_pred >= 0.5).astype(bool)
    y_pos = y_true.astype(bool)
    y_pred_neg = 1 - y_pred_pos
    y_neg = 1 - y_pos
    tp = np.sum(y_pos * y_pred_pos)
    tn = np.sum(y_neg * y_pred_neg)
    fp = np.sum(y_neg * y_pred_pos)
    fn = np.sum(y_pos * y_pred_neg)
    f1_score = (2 * tp + smooth) / (2 * tp + fn + fp + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    prec = (tp + smooth) / (tp + fp + smooth)  # precision
    rec = (tp + smooth) / (tp + fn + smooth)  # recall
    # tnr = (true_negative + smooth) / (true_negative + fp + smooth)
    return f1_score, iou, prec, rec


def calculate_iou_f1_score(y_true, y_pred):
    f1_score_list = []
    iou_list = []
    precision_list = []
    recall_list = []
    for k in range(len(y_true)):
        f1_score, iou, prec, rec = iou_f1_score(y_true[k, :, :, 0], y_pred[k, :, :, 0])
        f1_score_list.append(f1_score)
        iou_list.append(iou)
        precision_list.append(prec)
        recall_list.append(rec)
    iou, f1_score, precision, recall = (
        np.mean(np.array(iou_list)),
        np.mean(np.array(f1_score_list)),
        np.mean(np.array(precision_list)),
        np.mean(np.array(recall_list)),
    )
    return iou, f1_score
