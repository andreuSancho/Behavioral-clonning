# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 17:38:56 2017

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
input_path = PATH + "processedTrainData/"
output_path = PATH + "augmentedTrainData/"


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
        :param y_train: is the coded training labels (0, 1, ..., N).
        :param n_labels: is the number of distinct labels.
        :return: the histogram array.
    """
    # Compute the new histogram.
    histogram_train_label = np.zeros(n_labels, dtype="int32")
    for i in range(n_labels):
        indices = np.where(y_train == i)
        histogram_train_label[i] = len(indices[0])
    return(histogram_train_label)

def plotHistogram(histogram, binning_array, n_labels, angles=True):
    """
        It plots the histogram.
        :param histogram: is the computed histogram frequencies.
        :param binning_array: is the array containing the computed bins.
        :param n_labels: is the number of distinct labels.
        :param angles: it tells if we want to plot the angles or labels.
    """
     # Plot the class histogram.
    fig, ax = plt.subplots()
    step = binning_array[1] - binning_array[0]
    ind = np.arange(n_labels)  # The x locations for the groups.
    if angles:
        ax.bar(binning_array[0:n_labels], histogram, step, color='blue')
    else:
        ax.bar(ind[0:n_labels], histogram, 1, color='green')
    # add some text for labels, title and axes ticks
    ax.set_ylabel('Frequencies')
    ax.set_title('Angle distributions')
    if angles:
        ax.set_xticks(binning_array[0:n_labels])
        ax.set_xticklabels(binning_array[0:n_labels], rotation='vertical')
    else:
        ticks = [] 
        for i in range(n_labels):
            if i % 2 == 0:
                ticks.append(i)
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, rotation='vertical')
        
def grep(s, pattern):
    """
        Imitates grep.
        :param s: is the input string.
        :param pattern: is the pattern.
        :return: the grep answer.
    """
    return '\n'.join(re.findall(r'^.*%s.*?$'%pattern, s, flags=re.M))    

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

def setBinningSingleImage(dataset_output, binning_array):
    """
        It performs a binning in the given output steering data.
        :param dataset_output: the (single) output to encode.
        :param binning_array: the encoder.
        :return: the encoded labels.
    """
    label = 0
    for j in range(len(binning_array) - 1):
        if dataset_output >= binning_array[j] and dataset_output <= binning_array[j + 1]:
            label = j
            break
    return label

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
    if len(grep(data_log['path'][0], "^\s*IMG.+")) > 10:
        data_log['path'] = path_to_follow + data_log['path']
    dataset = []
    for f in data_log['path']:
        img = mpimg.imread(f)
        img = img.astype('uint8')
        dataset.append(img)
        del img
    dataset = np.array(dataset, dtype="uint8")
    labels = np.array(data_log['label'], dtype="uint8")    
    steering = np.array(data_log['steering'], dtype="float32")    
    return (dataset, steering, labels, data_log)
    
def generateBrigthness(image):
    """
        It generates new brigthness in images.
        Basen on Vivek's code https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.crqwrcuai
        :param image: the input image.
        :return: the new image with a randomized brightness.
    """
    brigth_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2HSV)
    brigth_image[:, :, 2] = brigth_image[:, :, 2 ] * 0.25 + np.random.uniform()
    np.max(brigth_image[:, :, 2])    
    brigth_image = cv2.cvtColor(brigth_image, cv2.COLOR_HSV2RGB)
    return brigth_image    

def translateImage(image, steering_angle=0.0, translation_range=20):
    """
        It simulates a translation of the original image. It adds an angle of 
        0.002 per displaced pixel.
        Stronlgy based on Vivke's code https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.crqwrcuai
        :param: image is the input image.
        :param: steering_angle is the image original steering.
        :param translation_range: is the amount of translation.
        :return: a tuple containing (1) the new image and (2) the new steering.
    """
    rows, cols, _ = image.shape
    tr_x = translation_range * np.random.uniform() - translation_range / 2
    steer_ang = steering_angle + tr_x / translation_range * 2 * 0.2
    tr_y = 40 * np.random.uniform() - 20
    Trans_M = np.float32([[1, 0, tr_x],
                          [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, Trans_M, (cols, rows))
    return (image_tr, steer_ang)
    
def addRandomShadow(image):
    """
        It adds a random shadow to the image.
        Stronlgy based on Vivke's code https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.crqwrcuai
        :param: image is the input image.
        :return: the shadowed image.
    """
    rows, cols, _ = image.shape
    top_y = cols * np.random.uniform()
    top_x = 0
    bot_x = rows
    bot_y = cols * np.random.uniform()
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    shadow_mask = 0 * image_hls[:, :, 1]
    X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]
    shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >=0)] = 1
    if np.random.randint(2) == 1:
        random_bright = 0.1 + np.random.uniform()
        cond1 = shadow_mask == 1
        cond0 = shadow_mask == 0
        if np.random.randint(2) == 1:
            image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1] * random_bright
        else:
            image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0] * random_bright    
    shadowImage = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)
    return shadowImage
    
def flipImage(image, steering):
    """
        It flips the given image / steering.
        :param image: is the input image tensor.
        :param steering: is the original steering.
        :return: a valid flipped image and steering.
    """
    img = cv2.flip(image.copy(), 1)
    s = -steering
    if s > 1.0:
        s = 1.0
    elif s < -1.0:
        s = -1.0
    return (img, s)

def changeTone(image):
    """
        It modifies the tonalities of input images. It does it by (1) substracting
        the color mean per channel and (2) adding the new plane color.
        of a plain color image to the original one.
        :param image: the input image.
        :return: the modified image.
    """
    # Randomly create a filter image of 3x3x3 px.
    r = np.ones(9, dtype="uint8")
    g = np.ones(9, dtype="uint8")
    b = np.ones(9, dtype="uint8")
    r = r.reshape(3, 3) * np.random.randint(256)
    g = g.reshape(3, 3) * np.random.randint(256)
    b = b.reshape(3, 3) * np.random.randint(256)
    source_img = cv2.merge([r, g, b]) 
    source_img = cv2.cvtColor(source_img, cv2.COLOR_RGB2LAB).astype("float32")
    del r, g, b
    # Transform our target image into a LAB color space.
    target_img = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2LAB).astype("float32")
    # Scale the target image.
    target_img[:, :, 0] -= np.mean(target_img[:, :, 0])
    target_img[:, :, 1] -= np.mean(target_img[:, :, 1])
    target_img[:, :, 2] -= np.mean(target_img[:, :, 2])
    # Add the filter color.
    target_img[:, :, 0] += np.mean(source_img[:, :, 0])
    target_img[:, :, 1] += np.mean(source_img[:, :, 1])
    target_img[:, :, 2] += np.mean(source_img[:, :, 2]) 
    # Clip.
    target_img[:, :, 0] = np.clip(target_img[:, :, 0], 0, 255)
    target_img[:, :, 1] = np.clip(target_img[:, :, 1], 0, 255)
    target_img[:, :, 2] = np.clip(target_img[:, :, 2], 0, 255)
    # Retransform to RGB.
    target_img = cv2.cvtColor(target_img.astype("uint8"), cv2.COLOR_LAB2RGB)
    return target_img   

def randomRotation(image, img_steering):
    """
        It adds a random rotation to an image. Note that a normalized angle of
        0.25 corresponds to approx. 6 degrees. So we use this information to
        rotate accordingly and generate the new steering angle.
        So, to have a [-1, 1] normalized angle we do: 0.04167 * angle_degrees,
        and to have a degree angle we do: 24 * angle_norm.
        :param image: is the input tensor.
        :param img_steering: is the steering data array.
        :return: the procesed image plus its new steering.
    """
    rows, cols, _ = image.shape
    rnd_term = 0.0001 + np.random.uniform()
    new_steering = img_steering
    if new_steering < 0.0:
        new_steering -= rnd_term 
        if new_steering < -1.0:
            new_steering = -1.0
    elif new_steering >= 0.0:    
        new_steering += rnd_term 
        if new_steering > 1.0:
            new_steering = 1.0
    # We've to be careful and not "over-rotate" the image! If it already got a
    # previous rotation, we have to adjust the remaining!
    rot_angle = 24 * (new_steering - img_steering)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rot_angle, 1)
    rotated_img = cv2.warpAffine(image.copy(), M, (cols, rows))
    return rotated_img, new_steering 
    
def augmentImages(image, steering, threshold=20, translate_px=32):
    """
        It augments images: given an image and its steering, it generates 
        a copy of the input and apply some transformation.
        :param image: is the input tensor.
        :param steering: is the steering data array.
        :param threshold: is the minimum amount of difference for bringhness/shadow.
        :param translate_px: maximum number of pixels for image translation.
        :return: the new training data, with the corresponding new steering
            data.
    """
    new_X = None
    new_s = None
    correctly_gen = False
    while not correctly_gen:
        option = np.random.randint(6)
        if option == 0: # Augment the brigthness.
            img = generateBrigthness(image)
            residual = np.abs(np.mean(np.abs(img.copy()) - np.abs(image.copy())))
            if residual >= threshold:
                new_X = img.copy()
                new_s = steering
                del img
                correctly_gen = True
        elif option == 1: # Augment the angle.
            img, s = translateImage(image, steering, translate_px)
            if s > 1.0:
                s = 1.0
            elif s < -1.0:
                s = -1.0
            new_X = img.copy()
            new_s = s
            del img
            correctly_gen = True
        elif option == 2: # Augment the shadows.
            img = addRandomShadow(image)
            residual = np.abs(np.mean(np.abs(img.copy()) - np.abs(image.copy())))
            if residual >= threshold:
                new_X = img.copy()
                new_s = steering
                del img
                correctly_gen = True
        elif option == 3: # Augment the tone of images.
            trial = np.random.uniform()
            if trial > 0.3:
                img = changeTone(image)
                new_X = img.copy()
                new_s = steering
                del img
                correctly_gen = True
        elif option == 4: # Augment by fliping the image.
            trial = np.random.uniform()
            if trial > 0.5:
                img, s = flipImage(image, steering)
                new_X = img.copy()
                new_s = s
                del img
                correctly_gen = True
        elif option == 5: # Augment with a random rotation.
            trial = np.random.uniform()
            if trial > 0.4:
                img, s = randomRotation(image, steering)
                new_X = img.copy()
                new_s = s
                del img
                correctly_gen = True
    return(new_X, new_s)
 
def generateBalancedTrainData(train_data, steering, histogram, binning_array, threshold=1000, upper_bound=3000):
    """
        It generates training data by augmenting existing one. This code has been
        adapted from Vivek's: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.crqwrcuai
        :param train_data: is the original training data tensor.
        :param steering: is the original steering.
        :param histogram: is the histogram to follow.
        :param binning_array: is array which contains the binnings.
        :param threshold: is the threshold for adding low-frequent images.
        :param upper_bound: is the maximum amount of cases in a bucket.
        :return: a batch of new images and steerings.
    """
    batch_images = []
    batch_steering = []
    y_data = setBinning(steering, binning_array)
    num_cases = len(histogram)
    print(" > Starting the augmentation process.")
    for i in range(num_cases):
        print("  > Case", str(i + 1) , "out of", num_cases)
        to_add = upper_bound - histogram[i]
        if to_add > 0:
            indices = np.where(y_data == i)[0]
            done = False
            counter = 0
            while not done:
                select = np.random.randint(indices.shape[0])
                i_rnd = indices[select]
                new_X, new_s = augmentImages(train_data[i_rnd], steering[i_rnd], 
                                 threshold=24, translate_px=20)
                if new_s == steering[i_rnd]: # No change in angle.
                    batch_images.append(new_X)
                    batch_steering.append(new_s)
                    counter += 1
                    if counter >= threshold:
                        done = True
                else: # Protect against overpopulating extremes.
                    new_bin = setBinningSingleImage(new_s, binning_array)
                    if i != 0 and i != (num_cases - 1):
                        if new_bin != 0 and new_bin != (num_cases - 1):
                            batch_images.append(new_X)
                            batch_steering.append(new_s)
                            counter += 1
                            if counter >= threshold:
                                done = True
                    else: # We're in the two extremes.
                        batch_images.append(new_X)
                        batch_steering.append(new_s)
                        counter += 1
                        if counter >= threshold:
                            done = True
    print(" > Finishing the augmentation process.")
    batch_images = np.array(batch_images, dtype="uint8")
    batch_steering = np.array(batch_steering, dtype="float32")
    return (batch_images, batch_steering)

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
        if histo[i] >= thresh: 
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
            if trial >= 0.8:
                new_X.append(train_data[i])
                new_y.append(y_train[i])
                new_s.append(steering[i])
    new_y = np.array(new_y)
    new_X = np.array(new_X, dtype="uint8")
    new_s = np.array(new_s, dtype="float32")
    return(new_X, new_y, new_s)    

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
train_data, steering, labels, _ = loadTrainingData(input_path)
# Crop the extremes.
train_data, labels, steering = cropMajority(train_data, steering, n_labels, binning_array, thresh=2000)
histo = generateHistogram(labels, n_labels)
plotHistogram(histo, binning_array, n_labels, False)
# Augment the data. Warning: very slow and memory consuming code!
X, s = generateBalancedTrainData(train_data, steering, histo, binning_array, threshold=300)
y = setBinning(s, binning_array) 
histo = generateHistogram(y, n_labels)
plotHistogram(histo, binning_array, n_labels, True)
# Join them together!
s = np.concatenate((steering, s))
del steering
X = np.concatenate((train_data, X))
del train_data, labels
y = setBinning(s, binning_array) 
histo = generateHistogram(y, n_labels)
plotHistogram(histo, binning_array, n_labels, True)
# Store the data.
paths=[]
for i in range(len(X)):
    rgb_img = cv2.cvtColor(X[i], cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path + "IMG/img_" + str(y[i]) + "_No_"+ str(i) + ".jpg", rgb_img)
    paths.append("IMG/img_" + str(y[i]) + "_No_"+ str(i) + ".jpg")
paths = np.array(paths)
df = pd.DataFrame({"label": y, "steering": s, "path": paths})
df.to_csv(output_path + "steering.csv", index=False)