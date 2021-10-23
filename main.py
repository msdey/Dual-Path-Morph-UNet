from model import *
from utils.preprocessing import *
from utils.loss import *
from utils.inference import *
import argparse
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K


parser = argparse.ArgumentParser(description="Dual Path Morph UNet")
group = parser.add_mutually_exclusive_group(required=True)

group.add_argument("--train", type=bool, help="Train the model")
group.add_argument("--test", type=bool, help="Test the model")

parser.add_argument(
    "-lr", "--LR", type=float, help="Learning rate of model", default=3e-4
)
parser.add_argument(
    "-batch", "--BATCH_SIZE", type=int, help="Batch size", default=8
)
parser.add_argument(
    "-epoch", "--EPOCHS", type=int, help="Number of epochs", default=120
)
parser.add_argument(
    "-levels", "--NUM_LEVELS", type=int, help="Number of thresholds", default=256,
)
parser.add_argument(
    "-preprocess", "--PREPROCESS", type=bool, help="Data is preprocessed ?", default=True,
)

args = parser.parse_args()

LR = args.LR
BATCH_SIZE = args.BATCH_SIZE
EPOCHS = args.EPOCHS
NUM_LEVELS = args.NUM_LEVELS
RESIZED = args.RESIZED
PREPROCESS = args.PREPROCESS
LOSS = dice_loss


def train_model(
    dataset="roads",
    preprocess=PREPROCESS,
    load_weights=None,
    lr=LR,
    loss=LOSS,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
):

    if preprocess is False:
        image_patches(type="train", dataset="roads")
        image_patches(type="valid", dataset="roads")
        image_patches(type="train", dataset="buildings")
        image_patches(type="valid", dataset="buildings")

    train_gen, valid_gen = data_generator(dataset=dataset, batch_size=batch_size)
    train_samples = len(os.listdir("./resized_data/" + str(dataset) + "/train/sat/"))
    valid_samples = len(os.listdir("./resized_data/" + str(dataset) + "/valid/sat/"))

    saved_weights = str(dataset) + "_weights.h5"
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=40)
    mc = ModelCheckpoint(
        saved_weights, monitor="val_loss", mode="min", verbose=1, save_best_only=True
    )

    model = DPM_UNet()
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=loss,
        metrics=[dice, precision, recall],
    )

    if load_weights is not None:
        model.load_weights(saved_weights)

    model.fit_generator(
        train_gen,
        validation_data=valid_gen,
        steps_per_epoch=train_samples / batch_size,
        validation_steps=valid_samples / batch_size,
        epochs=epochs,
        callbacks=[es, mc],
    )


def test_model(dataset="roads", preprocess=PREPROCESS, num_levels=NUM_LEVELS, lr=LR):
    if dataset != "roads" or dataset != "buildings":
        raise ValueError("dataset can have value either of 'roads' or 'buildings'")

    if preprocess is False:
        image_patches(type="test", dataset="roads")
        image_patches(type="test", dataset="buildings")

    saved_weights = str(dataset) + "_weights.h5"
    model = DPM_UNet()
    model.compile(optimizer=keras.optimizers.Adam(lr))
    model.load_weights(saved_weights)

    gt_img, pred_img = make_predictions(model, dataset)
    (
        bep,
        relaxed_bep,
        precision,
        recall,
        relaxed_precision,
        relaxed_recall,
    ) = calculate_metrics(y_true=gt_img, y_pred=pred_img, num_levels=num_levels)
    iou, f1_score = calculate_iou_f1_score(gt_img, pred_img)
    print(
        f" IoU : {round(iou, 2)} \n F1_score : {round(f1_score, 2)} \n BEP : {round(bep, 2)} \n Relaxed BEP : {round(relaxed_bep, 2)}"
    )

    visualise_pr_curve(relaxed_precision, relaxed_recall, relaxed_bep, relaxed=True)
    visualise_pr_curve(precision, recall, bep, relaxed=False)


if __name__ == "__main__":
    if args.train:
        K.clear_session()
        train_model()
    elif args.test:
        test_model()
