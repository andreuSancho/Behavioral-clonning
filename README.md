# Behavioral-cloning

Self-driving cars are a hot topic that has attracted the attention of many industrial and scientific practitioners. The problem itself consist in teaching a car controller to drive by itself, mimicking human drivers. This repository contains the code for such purpose: clone the behavior of human drivers and generalize the driving. The car controller used for this project is a deep convolutional neural network; a model capable of learning from color images. We refer to it as SimpLeNet, and its main task is to correctly predict the steering angle of the car from input images in a purely end-to-end fashion. 

For fitting the model, roughly 100,000 images were taken from multiple runs in a simulator. The simulator has two distinct circuits: a training circuit, and a test circuit. The car can train only using the first one and the fundamental goal is to **generalize** the driving for the second circuit. The amount of training images was processed (mainly cropped, scaled and rotated) for generating a final batch of 54,893 images from which SimpLeNet trained. The main concern is that these data are extremely unbalanced, favoring very little steering angles (humans tend to drive smoothly without many sudden changes in the steering, thus favoring driving in nearly straight line). The challenging process was to preprocess the data for filtering this extreme bias towards low angles (that is: driving in straight line), generating synthetic cases in which larger angles are favored. The data augmentation was a **critical** step for generalizing the driving, and without it the car controller was unable to run properly (read *safely*) in the test circuit. It is important to highlight that SimpLeNet saw **only** training examples from the **first** circuit (the training circuit).

Recall that this is the third project in Udacity’s nanodegree program in self-driving car engineering.

## The data set

As data are the **critical** part in any data-science project, multiple run have been recorded in training mode (a few hours of human driving in the train circuit). As kindly suggested by Udacity mentors, the runs were recorded by wandering off and moving along the lane from one side to the other (as if the driver was drunk) and playing with data refinement (that is: new data is added where the car fails). The data itself are divided in two parts: camera images (center, left and right) and telemetry data (steering angle and throttle). 

The input images have the following dimensions: 160 x 320 x 3 pixels. In the first step we (1) crop images (pixels 16 to 144 are taken) vertically and (2) scale down images to 64 x 160 x 3 pixels. Notice that the **aspect ratio is conserved**. With this strategy RAM memory is saved without a significant impact on the performance.

![Figure 1: original training data.](/images/originalDS.png)

The main challenge of this data set is the huge level of unbalanced cases. Figure 2 shows the histogram depicting this issue. We see three particular normalized angles (-0.23, 0.0, and 0.23) that consume more than 80% of the training set (see figure below). Therefore, our primary task will be to get rid of these extreme skewness by (1) randomly removing cases with large frequencies, (2) add left/right camera images adding an extra angle to the actual steering, (3) using images with a throttle greater than a user-defined threshold, and finally (4) augmenting the data set by adding random shifts, modifying brightness, adding random shadows and modifying colors to existing images. Another filter used for this project is to use images that have a minimum amount of throttle (the rationale is that we do not want standing still images but actual driving actions). This amount is 0.25 normalized throttle (or more).
