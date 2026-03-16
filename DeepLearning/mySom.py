#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 09:37:38 2018

@author: Bryan
"""

# Self Organizing Map
# Make a Fraud Detector
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
# Columns are attributes of customers
os.chdir('/Users/Bryan/Documents/Programming/Deep_Learning_A_Z/Volume 2 - Unsupervised Deep Learning/Mega_Case_Study')

dataset = pd.read_csv('Credit_Card_Applications.csv')

# Y is customers who were approved vs customers who were not approved
# We are looking for those that "cheated"
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

######## Feature Scaling
# Use normalization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

####### Train the SOM
# Get implementation from another developer
# MiniSom 1.0
from minisom import MiniSom

# input_len = number of features in X
# sigma radius of different neightbors in grid

som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)

# Initialize the weights as small numbers close to zero
som.random_weights_init(X)

# Need to input X and number of iterations
som.train_random(X, 100)

### Looking for MID - MEAN INTERNEURON DISTANCE
# Winning node colored, larger MID, closer to white

##### Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show

# Bone - initialize the map
bone()

# Add the values of all the mean distances from the winning nodes
# Closer to white is outlier winning nodes
# Get the transpose of the matrix
pcolor(som.distance_map().T)
# Add the legend
colorbar()

# Find customers closest to the white nodes
# Those that "cheated" and got approval are more important than "cheated" and not approved
# Create markers for approval red = no, green = yes
# o for circle, s for square
markers = ['o', 's']
colors = ['r', 'g']

# loop through all customers and assign markers and colors
# use i as the index, and x as the vector of customers
for i, x in enumerate(X):
    # get the winning node
    w = som.winner(x)
    # place a colored marker on it
    # adding 0.5 centers it
    plot(w[0] + 0.5,
         w[1] + 0.5,
         # uses y value (0 or 1 to select the circle or square for yes, no)
         # do the same for the colors to get red and green
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()


# Get the list of the cheaters
# Which customers fell into white nodes
# Use the data from which the map was trained
mappings = som.win_map(X)

# Find coordinates of white grid 7,8 
# also 5,1 are also fraud
# concatenate arrays along the horizontal vectors of customers using the vertical axis
fraud = np.concatenate((mappings[(7,8)], mappings[(5,1)]), axis = 0)

# inverse the scaling
fraud = sc.inverse_transform(fraud)






