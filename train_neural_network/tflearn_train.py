#!/usr/bin/env python
__author__ = "Thaynara Silva"
__copyright__ = "Copyright 2018, Software Development Final Year Project"
__version__ = "1.0"
__date__ = "19/04/2018"

import glob
import numpy as np
from sklearn.model_selection import train_test_split
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

image_array = np.zeros((1,784), np.float32)
label_array = np.zeros((1,3), np.float32)
training_file = glob.glob('training_data/*.npz')

for single_npz in training_file:
    with np.load(single_npz) as data:
        images_temp = data['train']
        labels_temp = data['train_labels']
    image_array = np.vstack((image_array, images_temp))
    label_array = np.vstack((label_array, labels_temp))

X = image_array[1:, :]
Y = label_array[1:, :]

train, test, train_labels, test_labels = train_test_split(X, Y, test_size=0.09)

train = train.reshape([-1, 28, 28, 1])
test = test.reshape([-1, 28, 28, 1])

cnn = input_data(shape=[None, 28, 28, 1], name='input')

cnn = conv_2d(cnn, 32, 2, activation='relu')
cnn = max_pool_2d(cnn, 2)

cnn = conv_2d(cnn, 64, 2, activation='relu')
cnn = max_pool_2d(cnn, 2)

cnn = fully_connected(cnn, 28, activation='relu')
cnn = dropout(cnn, 0.8)

cnn = fully_connected(cnn, 3, activation='softmax')
cnn = regression(cnn, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(cnn)

model.fit({'input': train}, {'targets': train_labels}, n_epoch=10, validation_set=({'input': test}, {'targets': test_labels}), 
    snapshot_step=500, show_metric=True, run_id='mnist')

model.save('model/cnn_model.model')