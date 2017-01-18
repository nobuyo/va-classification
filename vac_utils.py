# -*- coding: utf-8 -*-
from __future__ import print_function

import random
import os
import cv2
import numpy as np
import math

IMG_SIZE = 128

def array_divide(arr, n):
    return [arr[x:x + n] for x in range(0, len(arr), n)]

# checking existence of the class, train and test lists
def exist_list(list_dir):
    exists = os.path.exists(os.path.join('.', list_dir, 'class*.lst')) \
             and os.path.exists(os.path.join('.', list_dir, 'train*.lst')) \
             and os.path.exists(os.path.join('.', list_dir, 'test*.lst'))
    return exists

# create and return the class, train and test lists
def create_list_with_division(data_dir, list_dir, slash):
    classes = os.listdir(os.path.join('.', data_dir))
    data_list = []
    for i, cls in enumerate(classes):
        dir_name = os.path.join('.', data_dir, cls)
        if os.path.isdir(dir_name):
            files = os.listdir(dir_name)
            for f in files:
                data_list.append(os.path.join('.', data_dir, cls, f))

    random.shuffle(data_list)
    data_chunk = array_divide(data_list, 4000)

    # data_chunk = array_divide(data_list, int(math.ceil(len(data_list)/4)))
    train_list = []
    test_list = []
    for x in range(0,len(data_chunk)):
        split_index = int(len(data_chunk[x]) * slash)
        train_list.append(data_chunk[x][split_index:])
        test_list.append(data_chunk[x][:split_index])

    try:
        os.mkdir(list_dir)
    except OSError:
        print('Directory ./{0} already exists.'.format(list_dir))

    f = open(os.path.join('.', list_dir, 'class.lst'), 'w')
    f.write('\n'.join(classes))
    f.close()

    for x in range(0,len(data_chunk)):
        f = open(os.path.join('.', list_dir, 'train'+ str(x) +'.lst'), 'w')
        f.write('\n'.join(train_list[x]))
        f.close()
        f = open(os.path.join('.', list_dir, 'test'+ str(x) +'.lst'), 'w')
        f.write('\n'.join(test_list[x]))
        f.close()
    return classes, train_list[0], test_list[0]

def create_list(data_dir, list_dir, slash):
    classes = os.listdir(os.path.join('.', data_dir))
    data_list = []
    for i, cls in enumerate(classes):
        dir_name = os.path.join('.', data_dir, cls)
        if os.path.isdir(dir_name):
            files = os.listdir(dir_name)
            for f in files:
                data_list.append(os.path.join('.', data_dir, cls, f))

    split_index = int(len(data_list) * slash)
    random.shuffle(data_list)
    train_list = data_list[split_index:]
    test_list = data_list[:split_index]
    try:
        os.mkdir(list_dir)
    except OSError:
        print('Directory ./{0} already exists.'.format(list_dir))
    f = open(os.path.join('.', list_dir, 'class.lst'), 'w')
    f.write('\n'.join(classes))
    f.close()
    f = open(os.path.join('.', list_dir, 'train.lst'), 'w')
    f.write('\n'.join(train_list))
    f.close()
    f = open(os.path.join('.', list_dir, 'test.lst'), 'w')
    f.write('\n'.join(test_list))
    f.close()
    return classes, train_list, test_list

# load the class, train and test lists
def load_lists(list_dir):
    f = open(os.path.join('.', list_dir, 'class.lst'), 'r')
    classes = f.read().split()
    f.close()
    f = open(os.path.join('.', list_dir, 'train.lst'), 'r')
    train_list = f.read().split()
    f.close()
    f = open(os.path.join('.', list_dir, 'test.lst'), 'r')
    test_list = f.read().split()
    f.close()
    return classes, train_list, test_list

def load_lists_with_division(list_dir, divnum):
    f = open(os.path.join('.', list_dir, 'class.lst'), 'r')
    classes = f.read().split()
    f.close()
    f = open(os.path.join('.', list_dir, 'train'+ str(divnum) +'.lst'), 'r')
    train_list = f.read().split()
    f.close()
    f = open(os.path.join('.', list_dir, 'test'+ str(divnum) +'.lst'), 'r')
    test_list = f.read().split()
    f.close()
    return classes, train_list, test_list

# load images and add labels
def load_images(classes, data_list):
    images = []
    labels = []
    num_classes = len(classes)
    for data in data_list:
        # print(data, end='')
        img = cv2.imread(data)
        if not img is None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img/255.0
            images.append(img)
            lbl = np.zeros(num_classes)
            lbl[classes.index(os.path.basename(os.path.dirname(data)))] = 1
            labels.append(lbl)
    return images, labels

