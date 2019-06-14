#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:24:54 2018

@author: Bryan
"""


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation
from keras import optimizers
from keras.layers.noise import AlphaDropout

img_size = 128
n_epoch = 150
batch_size_train = 32
batch_size_test = 32
optimizer = optimizers.Adam(lr=0.0005)

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (5, 5),kernel_initializer='lecun_normal',padding = 'same', input_shape = (3, img_size, img_size)))
classifier.add(Activation('selu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Adding a second convolutional layer
classifier.add(Conv2D(64, (3, 3),kernel_initializer='lecun_normal', padding = 'same'))
classifier.add(Activation('selu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#classifier.add(AlphaDropout(0.2))

# Adding a third convolutional layer
classifier.add(Conv2D(128, (3, 3),kernel_initializer='lecun_normal',padding = 'same'))
classifier.add(Activation('selu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#classifier.add(AlphaDropout(0.1))

# Adding a fourth convolutional layer
classifier.add(Conv2D(256, (3, 3),kernel_initializer='lecun_normal', padding = 'same'))
classifier.add(Activation('selu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#classifier.add(AlphaDropout(0.05))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128,kernel_initializer='lecun_normal'))
classifier.add(Activation('selu'))
classifier.add(AlphaDropout(0.2))

classifier.add(Dense(units = 256,kernel_initializer='lecun_normal'))
classifier.add(Activation('selu'))
classifier.add(AlphaDropout(0.2))


#output layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

data_gen_args = dict(rescale = 1./255,
                     shear_range = 0.2,
                     zoom_range = 0.2,
                     height_shift_range = 0.2,
                     width_shift_range = 0.2,
                     horizontal_flip = True)

train_datagen = ImageDataGenerator(**data_gen_args)

test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (img_size, img_size),
                                                 batch_size = batch_size_train,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (img_size, img_size),
                                            batch_size = batch_size_test,
                                            class_mode = 'binary')

classifier_history_withBN = classifier.fit_generator(training_set,
                         steps_per_epoch = 8000/batch_size_train,
                         epochs = n_epoch,
                         validation_data = test_set,
                         validation_steps = 2000/batch_size_test)


#save weights
classifier.save_weights('WithSelueWithAlphaDropout_theano.h5')

# summarize history for accuracy
import matplotlib.pyplot as plt
plt.plot(classifier_history_withBN.history['acc'])
plt.plot(classifier_history_withBN.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()