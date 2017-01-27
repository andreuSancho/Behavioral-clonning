# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 10:06:21 2017

@author: a.sancho.asensio
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import re
import pandas as pd

if os.name == 'nt': # We're on the Windows machine.
    print(" > Loading paths for the Windows machine")
    PATH = "C:/Users/a.sancho.asensio/Documents/PaperWork/nanodegree/git/simulator-windows-64/"
else: # Linux/MAC machine.
    print(" > Loading paths for the Linux machine")
    PATH = "/home/andreu/nanodegree/simulator-linux/"
train_path_1 = PATH + "Keyboard_data1/"
train_path_2 = PATH + "Keyboard_data2/"
train_path_3 = PATH + "Keyboard_data3/"
train_path_4 = PATH + "Keyboard_data4/"
train_path_5 = PATH + "Keyboard_data5/"
train_path_6 = PATH + "Keyboard_data6/"
output_path = PATH + "processedTrainData/"
augmented_path = PATH + "augmentedTrainData/"

def plot(image):
    """
        It plots an image, independently of its number of channels.
        :param image: the image to plot.
    """
    if len(image.shape) == 2: # Grayscale
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)
    plt.axis('off')

def generateHistogram(y_train, n_labels):
    """
        It plots a histogram with the class distribution.
        :param y_train: the coded training labels (0, 1, ..., N).
        :param n_labels: the number of distinct labels.
        :return: the histogram array.
    """
    # Compute the new histogram.
    histogram_train_label = np.zeros(n_labels, dtype="int32")
    for i in range(n_labels):
        indices = np.where(y_train == i)
        histogram_train_label[i] = len(indices[0])
    return(histogram_train_label)

def plotHistogram(histogram, binning_array):
    """
        It plots the histogram.
        :param histogram: the computed histogram frequencies.
        :param binning_array: the array containing the computed bins.
    """
     # Plot the class histogram.
    fig, ax = plt.subplots()
    step = binning_array[1] - binning_array[0]
    ax.bar(binning_array[0:n_labels], histogram, step, color='blue')
    # add some text for labels, title and axes ticks
    ax.set_ylabel('Frequencies')
    ax.set_title('Angle distributions')
    ax.set_xticks(binning_array)
    ax.set_xticklabels(binning_array, rotation='vertical')
    
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
        Also, the output is in the CSV file "driving_log.csv". Images have the
        following shape: (64, 64, 3).
        :param path_to_follow: is the full path where the images are placed.
        :return: a list with (1) a numpy array with the images in RGB color,
             (2) a numpy array with the steering angle, and (3) the data log.
    """
    data_path = os.path.join(path_to_follow, "*.csv")
    files = glob.glob(data_path)
    data_log = pd.read_csv(files[0])
    # Check special case of relative paths...
    if len(grep(data_log['center'][0], "^\s*IMG.+")) > 10:
        data_log['center'] = path_to_follow + data_log['center']
    if len(grep(data_log['left'][0], "^\s*IMG.+")) > 10:
            data_log['left'] = path_to_follow + data_log['left']
    if len(grep(data_log['right'][0], "^\s*IMG.+")) > 10:
            data_log['right'] = path_to_follow + data_log['right']
    dataset = []
    steering = []    
    i = 0
    for f in data_log['center']:
        if data_log['throttle'][i] > 0.25:
            img = mpimg.imread(f)
            img = img[16:144, :, :] # Crop the image removing useless areas...
            img = cv2.resize(img, (160, 64), interpolation=cv2.INTER_AREA)
            dataset.append(img)
            steering.append(data_log['steering'][i])
        i +=1
    i = 0
    for f in data_log['left']:
        if data_log['throttle'][i] > 0.25:
            f = f.lstrip(' ')
            img = mpimg.imread(f)
            M = cv2.getRotationMatrix2D((160, 80), 6, 1)
            rotated_img = cv2.warpAffine(img.copy(), M, (320, 160))
            rotated_img = rotated_img[16:144, :, :] # Crop the image removing useless areas...
            rotated_img = cv2.resize(rotated_img, (160, 64), interpolation=cv2.INTER_AREA)
            dataset.append(rotated_img)
            del img, rotated_img, M
            tmp = data_log['steering'][i] + 0.25
            if tmp > 1.0:
                tmp = 1.0
            steering.append(tmp) # Add an extra angle...
        i +=1
    i = 0
    for f in data_log['right']:
        if data_log['throttle'][i] > 0.25:
            f = f.lstrip(' ')
            img = mpimg.imread(f)
            M = cv2.getRotationMatrix2D((160, 80), -6, 1)
            rotated_img = cv2.warpAffine(img.copy(), M, (320, 160))
            rotated_img = rotated_img[16:144, :, :] # Crop the image removing useless areas...
            rotated_img = cv2.resize(rotated_img, (160, 64), interpolation=cv2.INTER_AREA)
            dataset.append(rotated_img)
            del img, rotated_img, M
            tmp = data_log['steering'][i] - 0.25
            if tmp < -1.0:
                tmp = -1.0
            steering.append(tmp) # Substract an extra angle...
        i +=1
    dataset = np.array(dataset, dtype="uint8")
    steering = np.array(steering, dtype="float32")   
    return (dataset, steering, data_log)

def setBinning(dataset_output, binning_array):
    """
        It performs a binning in the data set output.
        :param dataset_output: the output to encode.
        :param binning_array: the encoder.
        :return: the encoded labels.
    """
    labels = np.zeros(len(dataset_output), dtype="uint8")
    for i in range(len(dataset_output)):
        for j in range(len(binning_array) - 1):
            if dataset_output[i] >= binning_array[j] and dataset_output[i] <= binning_array[j + 1]:
                labels[i] = j
                break
            
    return labels
   
def cropMajority(train_data, steering, n_labels, binning_array, thresh=500):
    """
        It crops those examples that are extremelly invalanced.
        :param train_data: is the training data bitch.
        :param steering: is the angle of the steer wheel.
        :param n_labels: is the number of bins.
        :param binning_array: is the array with the binnings.
        :param thresh: is the maximum threshold for cropping.
        :return: the new training, bins, and steering.
    """
    y_train = setBinning(steering, binning_array) 
    histo = generateHistogram(y_train, n_labels)
    new_X = []
    new_y = []
    new_s = []
    # Seek the frequency of each label in the data using the histogram.
    freq_array = []
    for i in range(n_labels):
        if histo[i] > thresh: 
            freq_array.append(i)
    freq_array = np.array(freq_array)
    # Locate those angles which fall into the index_array.
    index_array = []
    for i in range(len(train_data)):
        index = np.where(freq_array == y_train[i])[0]
        if len(index) > 0:
            index_array.append(i)
    index_array = np.array(index_array)
    # Ignore much of such indices.
    for i in range(len(train_data)):
        index = np.where(index_array == i)[0]
        if len(index) == 0: # If not found, add the case to the training data.
            new_X.append(train_data[i]) # It is a minority case.
            new_y.append(y_train[i])
            new_s.append(steering[i])
        else: # Add this majority case at random.
            trial = np.random.uniform()
            if trial >= 0.75:
                new_X.append(train_data[i])
                new_y.append(y_train[i])
                new_s.append(steering[i])
    new_y = np.array(new_y)
    new_X = np.array(new_X, dtype="uint8")
    new_s = np.array(new_s, dtype="float32")
    return(new_X, new_y, new_s)     
    
def loadTrainingData(path_to_follow):
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
    data_log['original'] = data_log['path'].copy()
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
    data_log['path'] = data_log['original'].copy()
    del data_log['original']
    return (dataset, steering, labels, data_log)

# Do the binning for avoiding extremely invalanced cases.
n_bins = 256
n_labels = n_bins
min_bin = -1.0
max_bin = 1.0
step = (max_bin - min_bin) / n_bins
binning_array = np.ones(n_bins + 1, dtype="float32")
for i in range(n_bins):
    binning_array[i] = min_bin + i * step
# Load the data set.
train_data, steering, _ = loadData(train_path_1)
y_train = setBinning(steering, binning_array) 
#histo = generateHistogram(y_train, n_labels)
#plotHistogram(histo, binning_array)
# Crop!
X, y, s = cropMajority(train_data, steering, n_labels, binning_array, 1000)
del train_data, steering

# Open the original data set.
T_X, T_s, T_y, T_log =  loadTrainingData(output_path)
#histo = generateHistogram(T_y, n_labels)
#plotHistogram(histo, binning_array)

# Store the data.
paths=[]
j = len(T_log)
for i in range(len(X)):
    rgb_img = cv2.cvtColor(X[i], cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path + "IMG/img_" + str(y[i]) + "_No_"+ str(i + j + 1) + ".jpg", rgb_img)
    paths.append("IMG/img_" + str(y[i]) + "_No_"+ str(i + j + 1) + ".jpg")
paths = np.array(paths)

df = pd.DataFrame({"label": y, "steering": s, "path": paths})
new_df = T_log.append(df, ignore_index = True)
new_df.to_csv(output_path + "steering.csv", index=False)
del X, y, s, paths, df, y_train, new_df
del T_X, T_s, T_y, T_log
