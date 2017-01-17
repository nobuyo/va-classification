import os
import math
import random
import glob
import numpy as np
from scipy import misc
from PIL import Image
import cv2

INPUT_DIR='./face_images'
OUTPUT_DIR='./va-dataset'

def flip_left_right(image):
    return image[:, -1::-1]

def random_brightness(image, max_delta=63, seed=None):
    img = np.array(image)
    delta = np.random.uniform(-max_delta, max_delta)
    image = Image.fromarray(np.uint8(img + delta))
    return image

def random_contrast(image, lower, upper, seed=None):
    factor = np.random.uniform(-lower, upper)
    mean = (image[0] + image[1] + image[2]).astype(np.float32) / 3
    img = np.zeros(image.shape, np.float32)
    for i in range(0, 3):
        img[i] = (img[i] - mean) * factor + mean
    return img

def crop(image, name, crop_size, padding_size):
    (width, height) = image.shape
    cropped_images = []
    for i in xrange(0, width, padding_size):
        for j in xrange(0, height, padding_size):
            box = (i, j, i+crop_size, j+crop_size) #left, upper, right, lower
            cropped_name = name + '_' + str(i) + '_' + str(j) + '.jpg'
            cropped_image = image[i:i+crop_size, j:j+crop_size]
            resized_image = cv2.resize(cropped_image, (IMAGE_SIZE, IMAGE_SIZE))
            cropped_images.append(resized_image)

    return cropped_images

def whitening(img):
    img = img.astype(np.float32)
    d, w, h = img.shape
    num_pixels = d * w * h
    mean = img.mean()
    variance = np.mean(np.square(img)) - np.square(mean)
    stddev = np.sqrt(variance)
    min_stddev = 1.0 / np.sqrt(num_pixels)
    scale = stddev if stddev > min_stddev else min_stddev
    img -= mean
    img /= scale
    return img

# データ拡張
# data_numに指定した値になるまで「左右反転」「輝度の変更」「コントラストの変更」「切り抜き」する
def data_augmentation(image_files, data_num):
    image_list = []
    file_num = len(image_files)

    for image_file in image_files:
        image_list.append(misc.imread(image_file))

    if file_num >= data_num:
        return image_list

    # flip left right
    random.shuffle(image_list)
    for image in image_list:
        flipped_image = flip_left_right(image)
        image_list.append(flipped_image)
        if len(image_list) == data_num:
            return image_list

    # random brightness
    random.shuffle(image_list)
    for image in image_list:
        brightness_image = random_brightness(image)
        image_list.append(brightness_image)
        if len(image_list) == data_num:
            return image_list

    # random contrast
    random.shuffle(image_list)
    for image in image_list:
        contrast_image = random_contrast(image)
        image_list.append(contrast_image)
        if len(image_list) == data_num:
            return image_list

    # cropping
    random.shuffle(image_list)
    image_list.clear()
    cropped_size = int(IMAGE_SIZE * random.uniform(0.65, 0.85))
    padding_size = IMAGE_SIZE - cropped_size
    for image in image_list:
        cropped_image_list = crop(image, 'image', cropped_size, padding_size)
        for cropped_image in cropped_image_list:
            image_list.append(cropped_image)
            if len(image_list) == data_num:
                return image_list

    return image_list


dir_list = os.listdir(INPUT_DIR)

for dir in dir_list:
    image_files = glob.glob(os.path.join(INPUT_DIR, dir, "*.jpg"))
    if len(image_files) == 0:
        continue

    image_list = data_augmentation(image_files, 500)
    out_dir = os.path.join(OUTPUT_DIR, dir_name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for i, image in enumerate(image_list):
        image = whitening(image)
        misc.imsave(os.path.join(OUTPUT_DIR, dir, str(i) + '.jpg'), image)
