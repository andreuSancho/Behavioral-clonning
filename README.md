# Behavioral-cloning

Self-driving cars are a hot topic that has attracted the attention of many industrial and scientific practitioners. The problem itself consist in teaching a car controller to drive by itself, mimicking human drivers. This repository contains the code for such purpose: clone the behavior of human drivers and generalize the driving. The car controller used for this project is a deep convolutional neural network; a model capable of learning from color images. We refer to it as SimpLeNet, and its main task is to correctly predict the steering angle of the car from input images in a purely end-to-end fashion. 

For fitting the model, roughly 100,000 images were taken from multiple runs in a simulator. The simulator has two distinct circuits: a training circuit, and a test circuit. The car can train only using the first one and the fundamental goal is to **generalize** the driving for the second circuit. The amount of training images was processed (mainly cropped, scaled and rotated) for generating a final batch of 54,893 images from which SimpLeNet trained. The main concern is that these data are extremely unbalanced, favoring very little steering angles (humans tend to drive smoothly without many sudden changes in the steering, thus favoring driving in nearly straight line). The challenging process was to preprocess the data for filtering this extreme bias towards low angles (that is: driving in straight line), generating synthetic cases in which larger angles are favored. The data augmentation was a **critical** step for generalizing the driving, and without it the car controller was unable to run properly (read *safely*) in the test circuit. It is important to highlight that SimpLeNet saw **only** training examples from the **first** circuit (the training circuit).

Recall that this is the third project in Udacity’s nanodegree program in self-driving car engineering.

## The data set

As data are the **critical** part in any data-science project, multiple run have been recorded in training mode (a few hours of human driving in the train circuit). As kindly suggested by Udacity mentors, the runs were recorded by wandering off and moving along the lane from one side to the other (as if the driver was drunk) and playing with data refinement (that is: new data is added where the car fails). The data itself are divided in two parts: camera images (center, left and right) and telemetry data (steering angle and throttle). 

The input images have the following dimensions: 160 x 320 x 3 pixels. In the first step we (1) crop images (pixels 16 to 144 are taken) vertically and (2) scale down images to 64 x 160 x 3 pixels. Notice that the **aspect ratio is conserved**. With this strategy RAM memory is saved without a significant impact on the performance.

![Figure 1: original training data, no processing done. Image shapes are of 160 x 320 x 3 pixels.](https://github.com/andreuSancho/P3-behavioral-cloning/blob/master/images/originalDS.png)

The main challenge of this data set is the huge level of unbalanced cases. Figure 2 shows the histogram depicting this issue. We see three particular normalized angles (-0.23, 0.0, and 0.23) that consume more than 80% of the training set (see figure below). Therefore, our primary task will be to get rid of these extreme skewness by (1) randomly removing cases with large frequencies, (2) add left/right camera images adding an extra angle to the actual steering, (3) using images with a throttle greater than a user-defined threshold, and finally (4) augmenting the data set by adding random shifts, modifying brightness, adding random shadows and modifying colors to existing images. Another filter used for this project is to use images that have a minimum amount of throttle (the rationale is that we do not want standing still images but actual driving actions). This amount is 0.25 normalized throttle (or more).

![Figure 2: Histogram (256 bins) on one of the many batches of training data, showing how angles are distributed. It is clear that three angles are consuming more than the 80% of the total data samples.](https://github.com/andreuSancho/P3-behavioral-cloning/blob/master/images/originalDataDistribution.png)

A part from the standard driving and preprocess, a second stage of refinement was used. This refinement consists in recording small driving actions where the model fails in order to rectify the behavior. Let’s exemplify this in the training circuit: in the first versions of the modelling, the car tended to go through the off-road part of the track (so far this is fine), the problem was that it tended to crash in the final left curve for exiting the off-road part and reenter to the main road. To refine the data, multiple refinement recordings were done in this particular part of the track and inserted into the final data set.

## Data augmentation

The main task in this project is to heavily augment the training set to fit the deep model, allowing the generalization of driving. For this purpose, a very nice blog post from Vivek Yadav was carefully read and the recommendations taken into consideration. The address of the web site is the following: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.fczm5xgd7. His post was very useful and didactic. I want to thank Vivek for his great post as it helped a lot.

The transformations done to the data set were the following:

**Use of left and right cameras**. In order to obtain a richer data set, left and right camera images were used. As noted by Vivek, the original images were slightly rotated for telling the model to go to the center of the lane. A change of 0.25 in the normalized steering angle seems to correspond to approximately 6 degrees. This was determined empirically and later confirmed in Vivek's post. In this regard, left camera images were rotated 6 degrees and an amount of 0.25 were added to the steering angle, and right camera images were rotated -6 degrees and an amount of -0.25 were added to the steering angle.

![Figure 3: Left and right cameras.](https://github.com/andreuSancho/P3-behavioral-cloning/blob/master/images/processedDS.png)

**Random brightness augmentation and reduction**. Directly taken from Vivek’s, it adds a brightness component to images. This way the model can train with different brightness levels, simulating from sunny days to dark evenings. This transformation is a very straightforward way for generalizing lightning conditions. It works as follows: given an input RGB image, it is transformed into HSV (hue, saturation, and value) color scheme and then a random uniform value is added to the value component, changing the brightness of the image. Also, a minimum threshold is set to avoid generating images with a very similar look to the original one.
 
![Figure 4: Random brightness augmentation and reduction.](https://github.com/andreuSancho/P3-behavioral-cloning/blob/master/images/processedDS_brightness.png)

**Random image displacement**. Another direct way of augmenting the data consist in shifting the images vertically and horizontally, thus simulating movement towards left-right and up-down. The steering angle is modified accordingly. This transformation has been adapted from Vivek's version. The maximum amount of pixels to displace the image is a user-defined parameter used to control the displacement.

![Figure 5: Random image displacement.](https://github.com/andreuSancho/P3-behavioral-cloning/blob/master/images/processedDS_translations.png)

**Random shadows**. As the training set does not have many large shadows, this transformation randomly adds dark patches to the input images. This way we allow the model to generalize in very different conditions. This transformation has been adapted from Vivek's one. Again, a minimum threshold is set to avoid generating images with a very similar look to the original one.

![Figure 6: Random shadows.](https://github.com/andreuSancho/P3-behavioral-cloning/blob/master/images/processedDS_shadows.png)

**Random tone augmentation**. As color will change from one circuit to another (think for instance in the color of the road or in different atmosphere conditions), a nice way to increase the capacity of the car controller is to augment the data set by generating different color tones of the training data. This way, the model will see many different color conditions that will help in the generalization process. It works as follows: a small image (3 x 3 x 3) is generated and a solid color is randomly chosen. This image is called "filter". The filter will contain the new tone to be added to images. Afterwards, the original image is transformed into L\* a\* b\* color space and its mean per channel subtracted. Then, the mean tone of the filter is added to the image per channel. Finally, the image is transformed into RGB.

![Figure 7: Random tone augmentation.](https://github.com/andreuSancho/P3-behavioral-cloning/blob/master/images/processedDS_color.png)

**Random image flipping**. In order to generate more images with distinct angles, if the original steering angle is larger than a user-defined threshold, the image is flipped horizontally (that is: a mirror effect), and the angle is inverted. 

**Random image rotation**. Similarly to image flipping, to generate new angles images are taken and a small random angle is added, thus rotating the image accordingly by a small amount. This way new cases are added to the data set.

![Figure 8: Random tone augmentation.](https://github.com/andreuSancho/P3-behavioral-cloning/blob/master/images/processedDS_rotations.png)

The results are summarized in the following histogram (256 bins), which contains 172,638 data samples with *all* angles. Notice that very high frequencies are randomly removed to avoid a possible bias.

![Figure 9: Histogram of the augmented data set.](https://github.com/andreuSancho/P3-behavioral-cloning/blob/master/images/figure_fullTrainAugmented.png)

## Modelling a human driver

The problem of modelling a human driver has attracted the attention of Machine Learning practitioners as it is a very complex challenge with many implications such as in safety. Also, it is especially appealing in the case of industry and many top players are investing in this technology. In this section, the model developed for cloning a human driver is first described. Afterward, the training strategy is detailed.

### SimpLeNet architecture

As useful data is limited (even with data augmentation), the car controller has to be simple enough for avoiding high bias (that is: overfitting) and at the same time deep enough for generalizing properly. Taking these restrictions into account, and after carefully reading Nvidia’s paper (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) and Mishkin et al. paper (https://arxiv.org/abs/1606.02228), the final model resembles LeNet but adding some extra layers for improving the car's behavior. The model is depicted in what follows.

![Figure 10: SimpLeNet architecture.](https://github.com/andreuSancho/P3-behavioral-cloning/blob/master/images/SimpLeNet.png)

The **three** main parts are clearly depicted: (1) the **heading** of the network, which chooses the best color space, (2) the **main body** of the convnet, which is quite similar to LeNet, and (3) the **tailing** part, which performs the regression to predict the steering angle.

Notice that the Explonential Linear Unit (ELU) is extensively used as non-linearity --except for the first two (very) Leaky ReLU’s, with an alpha of 0.4 each. ELU's are a great non-linearity that perform competently in all kinds of scenarios. Mishkin et al. (2016) work recommend the use of such activation functions, which avoid the need of a much computationally expensive batch normalization process. Also note the extensive use of dropout layers. Experiments done with the herein presented data reflect the importance of having dropout layers in the fully connected layers as overfitting may be a serious issue in this problem. The standard value of 0.5 keep probability is the selected one. 

The number of units has been determined empirically following the simplicity / accuracy trade-off, and having overfitting as low as possible.

Notice the usage of three fully connected layers. Interestingly, the first version of SimpLeNet did not use any, but for regression problems such as this one, the use of fully connected layers improve the results over the 1x1 convolution equivalent. Most of the model’s parameters are in these layers.

SimpLeNet has about 1,281,602 parameters.

### The training process

The training process followed the standard of data science: from the original data, 90% of it were used for training and the remaining 10% for testing using a **2-fold cross-validation** scheme. Also, an extra **validation set** was generated independently of the training/testing data (that is: using a new driving data), in which the important part is to determine how well the model identifies left, right and center cases. For simplicity, this validation data contains only three *extreme* cases: (1) an image of a very high angle towards left, (2) an image of a very high angle towards right, and (3) an image of a purely centered angle. This way, a rough estimate of the performance of the trained model is obtained.

The model was trained for 300 epochs using the Adam optimizer. It was configured using a learning rate of 0.0001 and a batch size of 600 images with the random shuffle activated to avoid biases. These values were obtained experimentally. Other optimizers were tested (SGD and RMSprop), but the best results were obtained with Adam.

## Discussion
An end-to-end solution for autonomous driving has been developed and tested under a simulator. Being a complex task, the key insight lies in obtaining a good data set. This data set has to be large enough and contain as many distinct steering angles as possible for the model to generalize well. It is not an exaggeration to say that roughly 98% of the project is in the generation of the training set. The extreme skewness of the driving recordings has a huge impact on the model performance, no matter how complex the model is. Therefore, the principles of exploratory data analysis (in short: plot everything and focus on how data is distributed) were key to achieve good results.

The data augmentation was enough for the car controller to drive in both tracks. Recall that test track cases were not in the training data. However, SimpLeNet tends to wander-off the lane. This is a drawback of getting rid of the main straight angles. In order to overcome this, the initial filtering may be relaxed allowing a bit of bias towards driving in straight lines. This bias has to be carefully calibrated empirically.

As future work lines we may (1) use distinct models and ensemble them for obtaining a richer car controller, (2) use the already trained model to follow a reinforcement learning stage to fine tune the driving, and (3) use the trained model in a real-world RC car by adding a camera to the device.
