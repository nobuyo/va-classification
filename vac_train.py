# -*- coding: utf-8 -*-
from __future__ import print_function

from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import LambdaCallback, EarlyStopping
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
    parser.add_argument('--division', dest='divnum', default=0)
    args = parser.parse_args()
    return args

args = parse_args()

if vac_utils.exist_list(args.list_dir):
    print('Lists already exist in ./{0}. Use these lists.'.format(args.list_dir))
    classes, train_list, test_list = vac_utils.load_lists_with_division(args.list_dir, args.divnum)
else:
    print('Lists do not exist. Create list from ./{0}.'.format(args.data_dir))
    classes, train_list, test_list = vac_utils.create_list_with_division(args.data_dir, args.list_dir, SLASH)

train_image, train_label = vac_utils.load_images(classes, train_list)
test_image, test_label = vac_utils.load_images(classes, test_list)

# convert to numpy.array
x_train = np.asarray(train_image)
y_train = np.asarray(train_label)
x_test = np.asarray(test_image)
y_test = np.asarray(test_label)

print('train samples: ', len(x_train))
print('test samples: ', len(x_test))

NUM_CLASSES = len(classes)
BATCH_SIZE = 32
EPOCH = 100

# building the model
if args.my_model is None:
    print('building the model ...')
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='valid',
                            input_shape=x_train.shape[1:]))
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
# callback_early_stop     = EarlyStopping(monitor='val_acc', patience=15, verbose=0, mode='auto')
callback_save_epoch_end = LambdaCallback(on_epoch_end=model.save('epc_' + date_str + '.model'))

# training
hist = model.fit(x_train, y_train,
                 batch_size=BATCH_SIZE,
                 verbose=1,
                 nb_epoch=EPOCH,
                 validation_data=(x_test, y_test),
                 # callbacks=[callback_save_epoch_end, callback_early_stop])
                 callbacks=[callback_save_epoch_end])

# save model
model.save('vac' + str(args.divnum) + '.model')

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
