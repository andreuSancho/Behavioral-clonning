# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 12:22:44 2017

@author: a.sancho.asensio
"""

import argparse
import base64
import json
import re, sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import math
import pandas as pd
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Input, merge, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, ELU
import keras.backend as K

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

if os.name == 'nt': # We're on the Windows machine.
    print(" > Loading paths for the Windows machine")
    PATH = "C:/Users/a.sancho.asensio/Documents/PaperWork/nanodegree/git/simulator-windows-64/"
else: # Linux/MAC machine.
    print(" > Loading paths for the Linux machine")
    PATH = "/home/andreu/nanodegree/simulator-linux/"

g_steering = np.zeros(10, dtype="float32") # Global array containing the last steering angles.
sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    #old_angle = float(data["steering_angle"]) / 25.0 # We need to normalize!
    # The current throttle of the car
    #throttle = data["throttle"]
    # The current speed of the car
    #speed = float(data["speed"])
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image, dtype="uint8")
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    image_array = image_array[16:144, :, :] # Crop the image removing useless areas...
    image_array = cv2.resize(image_array, (160, 64), interpolation=cv2.INTER_AREA)
    transformed_image_array = image_array[None, :, :, :]
    prediction = float(model.predict(transformed_image_array, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 1.0
    # Filter the data.
    global g_steering
    final_steering = 0.9 * prediction + 0.1 * np.mean(g_steering)
    g_steering = np.roll(g_steering, 1)
    g_steering[0] = final_steering
    print("{:.3f}".format(final_steering), "{:.3f}".format(throttle))
    send_control(final_steering, throttle)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    # Set the seed and load the model.
    np.random.seed(1337)
    tf.set_random_seed(1337) # Tensorflow specific.
    with open(args.model, 'r') as jfile:
       model = model_from_json(jfile.read())
    model.compile("adam", "mse") 
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)
 
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)