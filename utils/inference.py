import time
import glob
from scipy.interpolate import interp1d
from PIL import Image
from utils.metrics import *
from utils.misc import *


def calculate_metrics(y_true, y_pred, num_levels):
    num_levels = 256
    threshold = list(set(map(lambda x: x / (num_levels - 1), np.arange(num_levels))))
    threshold.sort()  # sorting the threshold into ascending order

    pr_start_time = time.time()
    precision, recall = calculate_precision_recall(y_true, y_pred, threshold)
    precision.append(1)
    recall.append(0)
    pr_end_time = time.time()
    print(
        f"Computed precision & recall in {round( (pr_end_time - pr_start_time), 2 )} secs "
    )

    relaxed_pr_start_time = time.time()
    relaxed_precision, relaxed_recall = calculate_precision_recall(
        y_true, y_pred, threshold, relaxed=True
    )
    relaxed_precision.append(1)
    relaxed_recall.append(0)
    relaxed_pr_end_time = time.time()
    print(
        f"Computed precision & recall in {round((relaxed_pr_end_time - relaxed_pr_start_time), 2)} secs "
    )

    ip_func = interp1d(precision[:], recall[:], kind="quadratic")
    precision_ip = np.linspace(0, 1, 100)
    recall_ip = ip_func(precision_ip)
    bep = calculate_breakeven_point(precision_ip, recall_ip)

    relaxed_ip_func = interp1d(relaxed_precision[:], relaxed_recall[:], kind="linear")
    relaxed_precision_ip = np.linspace(0, 1, 1000)
    relaxed_recall_ip = relaxed_ip_func(relaxed_precision_ip)
    relaxed_bep = calculate_breakeven_point(relaxed_precision_ip, relaxed_recall_ip)

    return (
        bep,
        relaxed_bep,
        precision_ip,
        recall_ip,
        relaxed_precision_ip,
        relaxed_recall_ip,
    )


def make_predictions(model, dataset="roads"):
    sat_images_path = "./resized_data/" + str(dataset) + "/test/sat/*png"
    sat_files = glob.glob(sat_images_path)

    pred_img = []
    gt_img = []

    for sat_file in sat_files:
        map_file = sat_file.replace("sat", "map")

        sat_img = Image.open(sat_file)
        sat_img = normalise(sat_img)

        sat_img = sat_img[np.newaxis, ...]
        result = model.predict(sat_img)
        pred_img.append(result[0, :, :, :])

        map_img = Image.open(map_file)
        map_img = normalise(map_img)
        gt_img.append(map_img)

    gt_img = np.asarray(gt_img)
    pred_img = np.asarray(pred_img)

    return gt_img, pred_img


def visualise_pr_curve(precision, recall, bep, relaxed=False):
    plt.style.use("seaborn-whitegrid")
    plt.xlim(0.4, 1.0)
    plt.ylim(0.4, 1.0)

    if relaxed:
        type = "Relaxed "
    else:
        type = ""

    plt.plot(precision, recall, label=str(type) + "PR curve")
    plt.plot(bep, bep, "x", markersize=10, label=str(type) + "BEP")
    plt.legend()
    plt.xlabel(str(type) + "Precision")
    plt.ylabel(str(type) + "Recall")
    plt.title(str(type) + "Precision Recall Curve")
    plt.show()
