from utils import *
from PIL import Image
import numpy as np
import os
import glob
from keras.preprocessing.image import ImageDataGenerator

building_path = './data/buildings/'
road_path = './data/roads/'

def image_patches(image_size = (1024,1024), patch_size = (256,256), type = 'train', dataset = 'roads'):
    if dataset == 'roads':
        sat_images_path = road_path + str(type) + '/sat/'
        map_images_path = road_path + str(type) + '/map/'
    elif dataset == 'buildings':
        sat_images_path = building_path + str(type) + '/sat/'
        map_images_path = building_path + str(type) + '/map/'
    else :
        raise ValueError("dataset can have value either of 'roads' or 'buildings'")
    file_count = 0
    sat_files_list = glob.glob(sat_images_path + '/*tiff')
    for sat_file in sat_files_list:
        map_file = sat_file.replace('sat', 'map').replace('.tiff', '.tif')

        sat_img = Image.open(sat_file)
        sat_img = sat_img.resize(image_size)
        sat_img = np.asarray(sat_img)
        patch_number = 0
        for i in range(0, image_size[0], patch_size[0]):
            for j in range(0, image_size[1], patch_size[1]):
                patch_img = sat_img[j:j + patch_size[1], i:i + patch_size[0]]
                img_name = './resized_data/' + str(dataset) + '/' + str(type) + '/sat_' + str(file_count) + "_" + str(patch_number) + '.png'
                Image.fromarray(patch_img).save(img_name)
                patch_number += 1

        map_img = Image.open(map_file)
        map_img = map_img.resize(image_size)
        map_img = np.asarray(map_img)
        patch_number = 0
        for i in range(0, image_size[0], patch_size[0]):
            for j in range(0, image_size[1], patch_size[1]):
                if dataset == 'buildings':
                    patch_img = map_img[j:j + patch_size[1], i:i + patch_size[0], 0]
                elif dataset == 'roads':
                    patch_img = map_img[j:j + patch_size[1], i:i + patch_size[0]]
                img_name = './resized_data/' + str(dataset) + '/' + str(type) + '/map_' + str(file_count) + "_" + str(patch_number) + '.png'
                Image.fromarray(patch_img).save(img_name)
                patch_number += 1

        file_count += 1

def data_generator(batch_size=16, seed=16, dataset = 'roads'):
    if dataset == 'buildings':
        map_train_path = './resized_data/buildings/train/map/'
        sat_train_path = './resized_data/buildings/train/sat/'
        map_valid_path = './resized_data/buildings/valid/map/'
        sat_valid_path = './resized_data/buildings/valid/sat/'
    elif dataset == 'roads':
        map_train_path = './resized_data/roads/train/map/'
        sat_train_path = './resized_data/roads/train/sat/'
        map_valid_path = './resized_data/roads/valid/map/'
        sat_valid_path = './resized_data/roads/valid/sat/'
    else :
        raise ValueError("dataset can have value either of 'roads' or 'buildings'")

    data_gen_args = dict(rescale= 1./ 255, horizontal_flip=True, vertical_flip=True)
    datagen = ImageDataGenerator(**data_gen_args)

    sat_valid = datagen.flow_from_directory(sat_valid_path, class_mode=None, seed=seed, batch_size=batch_size)
    map_valid = datagen.flow_from_directory(map_valid_path, class_mode=None, seed=seed, batch_size=batch_size,
                                            color_mode='grayscale')
    sat_train = datagen.flow_from_directory(sat_train_path, class_mode=None, seed=seed, batch_size=batch_size)
    map_train = datagen.flow_from_directory(map_train_path, class_mode=None, seed=seed, batch_size=batch_size,
                                            color_mode='grayscale')

    valid_gen = zip(sat_valid, map_valid)
    train_gen = zip(sat_train, map_train)

    return train_gen, valid_gen



