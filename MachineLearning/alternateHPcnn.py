#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:17:56 2018

@author: Bryan
"""

# coding: utf-8
# In[1]:
# Convolutional Neural Network
# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
# Installing Tensorflow
# pip install tensorflow
# Installing Keras
# pip install --upgrade keras
# Part 1 - Building the CNN
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
# In[2]:
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))
# In[3]:
# Image dimensions
img_width, img_height = 150, 150 
# In[4]:
# Initialising the CNN
classifier = Sequential()
# Regulize
# classifier.add(Dropout(rate=0.2, input_shape = (128, 128, 3)))
# Step 1 - Adding a first convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (img_width, img_height, 3)))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a third convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a fourth convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dropout(rate=0.6))
classifier.add(Dense(units = 64, activation='relu'))
classifier.add(Dropout(rate=0.3))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# In[5]:
# Part 2 - Fitting the CNN to the images
# Preprocess
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (img_width, img_height),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (img_width, img_height),
                                            batch_size = 32,
                                            class_mode = 'binary')
# In[8]:
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 2000,
                         workers = 5)


classifier.save('model name')

# use val_acc to get cross validation score

