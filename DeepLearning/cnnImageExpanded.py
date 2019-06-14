#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 08:00:16 2017

@author: Bryan
"""

# convolutional neural networks


# load the data set
import os

os.chdir(os.path.normpath('/Users/Bryan/Documents/Programming/TrainingDataResources/Convolutional_Neural_Networks/'))

from keras.models import Sequential

# for images
from keras.layers import Convolution2D

# proceed to add pooling layers
from keras.layers import MaxPooling2D

# convert pooled feature maps to large feature vector
from keras.layers import Flatten


from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# step - 1 add the first layer - convolutional layer
# create 32 feature detectors of 3 x 3

classifier.add(Convolution2D(32, 3, 3, input_shape = (128, 128, 3), activation = 'relu'))


# step - 2 max pooling using 2 x 2 martrix

classifier.add(MaxPooling2D(pool_size = (2, 2)))

# add a second convolutional layer
# apply to pooled feature maps
# no input shape
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# adding a 3rd layer of 64 feature detactors rather than 32


# step - 3 flattening into a large single vector

classifier.add(Flatten())

# step - 4 full connnection create a hidden layer

classifier.add(Dense(output_dim = 128, activation = 'relu'))

# add the output layer
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# compile the model - CNN
# create stochastic gradient descent
# create a loss function
# specify a performance metric

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - fitting the CNN to the images
# perform data augmentation to create more images for training
# augments the amount of images for training
# reduce the likelihood of overfitting

from keras.preprocessing.image import ImageDataGenerator

# applies random transformations shear, zoom, flip
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# rescale the pixels of the images of the test set
# values rescaled from 0 - 255 to 0 - 1
test_datagen = ImageDataGenerator(rescale=1./255)

# set the directory
# target size is based on image dimensions
# class mode is binary: cat or dog
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

# samples per epoch is size of traning set
# epochs number of runs
# validation data set to testing
# num_val_samples = size of testing set

classifier.fit_generator(
        training_set,
        samples_per_epoch=8000,
        nb_epoch=25,
        validation_data=test_set,
        nb_val_samples=2000)









