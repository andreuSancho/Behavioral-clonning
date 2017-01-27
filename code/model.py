# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 12:22:44 2017

@author: a.sancho.asensio
"""

import argparse
import base64
import json
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import re, sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.ndimage import convolve
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input, merge, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, ELU
import keras.backend as K
import tensorflow as tf
tf.python.control_flow_ops = tf

if os.name == 'nt': # We're on the Windows machine.
    print(" > Loading paths for the Windows machine")
    PATH = "C:/Users/a.sancho.asensio/Documents/PaperWork/nanodegree/git/simulator-windows-64/"
else: # Linux/MAC machine.
    print(" > Loading paths for the Linux machine")
    PATH = "/home/andreu/nanodegree/simulator-linux/"
train_path = PATH + "processedTrainData/"
augmented_path = PATH + "augmentedTrainData/"
test_path = PATH + "processedTestData/"
validation_path = PATH + "validation/"

def grep(s, pattern):
    """
        Imitates grep.
        :param s: is the input string.
        :param pattern: is the pattern.
        :return: the grep answer.
    """
    return '\n'.join(re.findall(r'^.*%s.*?$'%pattern, s, flags=re.M)) 
    
def loadData(path_to_follow):
    """
        It loads the images, assuming that these are in the /IMG/ subfolder.
        Also, the output is in the CSV file "steering.csv".
        :param path_to_follow: is the full path where the images are placed.
        :return: a list with (1) a numpy array with the images in RGB color,
             (2) a numpy array with the steering angle, (3) a numpy array
             with the class label, and (4) the data logs.
    """
    data_path = os.path.join(path_to_follow, "*.csv")
    files = glob.glob(data_path)
    data_log = pd.read_csv(files[0])
    # Check special case of relative paths...
    if len(grep(data_log['path'][0], "^\s*IMG.+")) > 10:
        data_log['path'] = path_to_follow + data_log['path']
    dataset = []
    for f in data_log['path']:
        img = mpimg.imread(f)
        img = img.astype('uint8')
        dataset.append(img)
        del img
    dataset = np.array(dataset)
    labels = np.array(data_log['label'], dtype="uint8")    
    steering = np.array(data_log['steering'], dtype="float32")    
    return (dataset, steering, labels, data_log)

# Load the data set.
print(" > Loading the train set in dir", augmented_path)
train_data, train_steering, _, _ = loadData(augmented_path)
X_train, X_test, y_train, y_test = train_test_split(train_data, 
                                                    train_steering,
                                                    test_size=0.1, 
                                                    random_state=1337)
del train_data, train_steering
print(" > Loading the validation set.")
validation_data, validation_steering, validation_labels, _ = loadData(validation_path)
      
# Construct the model.    
def simpLeNet(input_shape, learning_rate=0.001):
    """
        It builds the simpleNet model.
        :param input_shape: shape of the input tensor.
        :param learning_rate: is the learning rate.
    """
    model = Sequential()
    # Start the normalization layers.
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(input_shape[1],
                                                             input_shape[2],
                                                             input_shape[3])))
    model.add(Lambda(lambda x: x * 2.0))
    # Start the mini-network for pre-processing color channels.
    model.add(Convolution2D(10, 1, 1, subsample=(1, 1), border_mode="same",
                            init="he_normal", dim_ordering="tf"))
    model.add(LeakyReLU(alpha=0.48))
    model.add(Convolution2D(3, 1, 1, subsample=(1, 1), border_mode="same",
                            init="he_normal", dim_ordering="tf"))
    model.add(LeakyReLU(alpha=0.48))
    model.add(MaxPooling2D((2, 3), border_mode='same'))
    # Start the image processing layers.
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same",
                            init="he_normal", dim_ordering="tf"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same",
                            init="he_normal", dim_ordering="tf"))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2), border_mode='same'))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="same",
                            init="he_normal", dim_ordering="tf"))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="same",
                            init="he_normal", dim_ordering="tf"))
    model.add(ELU())
    model.add(MaxPooling2D((2, 4), border_mode='same'))
    model.add(Dropout(0.5))
    # Start the regression net.
    model.add(Convolution2D(256, 1, 1, subsample=(1, 1), border_mode="same",
                            init="he_normal", dim_ordering="tf"))
    model.add(ELU())
    model.add(Dropout(0.4))
    model.add(Convolution2D(128, 1, 1, subsample=(1, 1), border_mode="same",
                            init="he_normal", dim_ordering="tf"))
    model.add(ELU())
    model.add(GlobalAveragePooling2D(dim_ordering='tf'))
    model.add(Dense(1, init="he_normal"))
    
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return(model)

def fitModel(train_set, train_labels, file_name, test_set, test_labels, 
             validation_data, validation_steering, 
             batch_size=128, n_epoch=5, learning_rate=0.1, seed=17):
    """
        It fits a model.
        :param train_set: is the training data tensor.
        :param train_labels: is the steering angle array.
        :param file_name: is the output name (with path) for the model.
        :param test_set: is the test set data tensor.
        :param test_labels: is the test output.
        :param validation_data: is the validation set data tensor.
        :param validation_steering: is the valiadtion output.
        :param batch_size: is the size of the mini-batch.
        :param n_epoch: is the number of epoch to train the model.
        :param learning_rate: is the learning rate.
        :param seed: is the random seed.
        :return: the trained model.
    """
    np.random.seed(seed)
    tf.set_random_seed(seed) # Tensorflow specific.
    input_shape = train_set.shape
    model = simpLeNet(input_shape, learning_rate)
    print(model.summary())
    model.fit(train_set, train_labels,
          batch_size=batch_size,
          nb_epoch=n_epoch,
          validation_data=(test_set, test_labels),
          shuffle=True)
    print(" > Checking at epoch =", str(n_epoch))
    img = validation_data[0]
    img = img[None, :, :, :]
    print("  > Left: ", model.predict(img, batch_size=1), " (it sould be ", 
          str(validation_steering[0]), ")")   
    img = validation_data[1]
    img = img[None, :, :, :]
    print("  > Right: ", model.predict(img, batch_size=1), " (it sould be ", 
          str(validation_steering[1]), ")")   
    img = validation_data[2]
    img = img[None, :, :, :]
    print("  > Center: ", model.predict(img, batch_size=1), " (it sould be ", 
          str(validation_steering[2]), ")") 
    # if file_name is provided, store the model.
    if file_name != None:
        model.save_weights(file_name + ".h5", overwrite=True)
        json_string = model.to_json()
        open((file_name + ".json"), 'w').write(json_string)
    return(model)

# Train the model.
model = fitModel(train_set=X_train, 
                 train_labels=y_train, 
                 file_name=PATH + "model",
                 test_set=X_test, 
                 test_labels=y_test,
                 validation_data=validation_data,
                 validation_steering=validation_steering,
                 batch_size=500, 
                 n_epoch=200, 
                 learning_rate=0.0001,
                 seed=1337)   
