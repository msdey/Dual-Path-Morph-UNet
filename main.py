import tensorflow as tf
from model import  *
from  preprocessing import *
from  utils import *
import argparse
import keras
from keras.callbacks import Callback,  EarlyStopping, ModelCheckpoint
from keras import backend as K

parser = argparse.ArgumentParser(description="Train model")
parser.add_argument('-lr', '--LR', type=float, metavar='', required=False, help='Learning rate of model')
parser.add_argument('-b', '--BATCH_SIZE', type=int, metavar='', required=False, help='Batch size')
parser.add_argument('-ep', '--EPOCHS', type=int, metavar='', required=False, help='Number of epochs')
parser.add_argument('train', type=str,  help='Training the model')
args = parser.parse_args()


LR = 5E-3
LOSS = dice_loss
BATCH_SIZE = 8
EPOCHS = 120


def train_model(dataset='roads', resized=True, load_weights = None, lr = LR, loss = LOSS, batch_size = BATCH_SIZE, epochs = EPOCHS):

    if resized is False:
        image_patches(type='train', dataset='roads')
        image_patches(type='test', dataset='roads')
        image_patches(type='valid', dataset='roads')
        image_patches(type='train', dataset ='buildings')
        image_patches(type='test', dataset='buildings')
        image_patches(type='valid', dataset='buildings')

    train_gen, valid_gen = data_generator(dataset = dataset, batch_size = batch_size)
    train_samples = len(os.listdir('./resized_data/' + str(dataset) + '/train/sat/'))
    valid_samples = len(os.listdir('./resized_data/' + str(dataset) + '/valid/sat/'))

    saved_weights = str(dataset) + '_weights.h5'
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40)
    mc = ModelCheckpoint(saved_weights, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    model = DPM_UNet()
    model.compile(optimizer= keras.optimizers.Adam(lr), loss=loss, metrics=[dice, prec, rec])

    if load_weights is not None:
        model.load_weights(saved_weights)

    model.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_samples/batch_size, validation_steps=valid_samples/ batch_size, epochs=epochs, callbacks=[es, mc])


if __name__ == "__main__":
    if args.train:
        K.clear_session()
        train_model()

