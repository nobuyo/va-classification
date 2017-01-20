# -*- coding: utf-8 -*-
from __future__ import print_function

from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
import vac_utils

SLASH = 0.2 # percentage of test(validation) data

# parsing arguments
def parse_args():
    parser = argparse.ArgumentParser(description='image classifier')
    parser.add_argument('--data', dest='data_dir', default='data')
    parser.add_argument('--list', dest='list_dir', default='list')
    parser.add_argument('--model', dest='my_model', default=None)
    args = parser.parse_args()
    return args

args = parse_args()

print('train samples: ', len(x_train))
print('test samples: ', len(x_test))

NUM_CLASSES = len(classes)
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCH = 2000

# building the model
if args.my_model is None:
    print('building the model ...')
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='valid',
                            input_shape=(IMAGE_SIZE,3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    # model.add(Dense(128))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))

    rmsplop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=rmsplop, metrics=['accuracy'])
else:
    print('rebuilding the model ...')
    model = load_model(args.my_model)

date_str = datetime.datetime.now().strftime('%Y%m%d%H%M')
callback_early_stop = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto');
callback_save_model = ModelCheckpoint('vac' + str(args.divnum) + '.model', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

# training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE)

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE)

model.fit_generator(
        train_generator,
        samples_per_epoch=3200,
        nb_epoch=EPOCH,
        validation_data=validation_generator,
        nb_val_samples=800)

# plot loss
print(hist)
loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

nb_epoch = len(loss)
fig, ax1 = plt.subplots()
ax1.plot(range(nb_epoch), loss, label='loss', color='b')
ax1.plot(range(nb_epoch), val_loss, label='val_loss', color='g')
leg = plt.legend(loc='upper left', fontsize=10)
leg.get_frame().set_alpha(0.5)
ax2 = ax1.twinx()
ax2.plot(range(nb_epoch), acc, label='acc', color='r')
ax2.plot(range(nb_epoch), val_acc, label='val_acc', color='m')
leg = plt.legend(loc='upper right', fontsize=10)
leg.get_frame().set_alpha(0.5)
plt.grid()
plt.xlabel('epoch')
plt.savefig('graph_' + date_str + '.png')
# plt.show()
