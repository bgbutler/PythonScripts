#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 10:23:50 2018

@author: Bryan
"""

# Combine an ANN with SoM

# Part 1 - The unsupervised deep learning branch

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

os.chdir('/Users/Bryan/Documents/Programming/Deep_Learning_A_Z/Volume 2 - Unsupervised Deep Learning/Mega_Case_Study')


# Import the dataset
os.chdir('/Users/Bryan/Documents/Programming/Deep_Learning_A_Z/Volume 2 - Unsupervised Deep Learning/Mega_Case_Study')
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

######## Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

####### Train the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(X, 100)


##### Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()


# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(1,2)], mappings[(2,1)]), axis = 0)
frauds = sc.inverse_transform(frauds)

# create the matrix of features
customers = dataset.iloc[:,1:].values

# create the dependent variable
is_fraud = np.zeros(len(dataset))

# loop through customers to see if the id is in the frauds
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1
    

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)


# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# have 15 features in customers
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))


# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size = 10, epochs = 2)


# Predicting the probability of frauds
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:,0:1].values, y_pred), axis = 1)

# sort by probability descending
y_pred = y_pred[y_pred[:, 1].argsort()]



