#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 09:13:58 2018

@author: Bryan

Implement LSTM on Google stock
"""

import os

dataFile = os.path.normpath("~/Documents/Programming/Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Recurrent_Neural_Networks/Google_Stock_Price_Train.csv")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Part - 1 Data Preprocessing
# Import the training set

# Want a np array not a single vector
# using 1:2 excludes column 2, but creates an array use .array to make np.array
dataset_train = pd.read_csv(dataFile)
training_set = dataset_train.iloc[:,1:2].values

# Normalization - subtract min (x-min)/max - min
# Standarization - subtract mean (x - mean)/StdDev

# Use Normalization for Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

# Number of time steps
# What needs to be remembered, critical to prevent overfitting
# Create datastructure with 60 timesteps and 1 output
# Looks at 60 lags of time (3 months)
X_train = []
y_train = []

for i in range(60, 1258):
    # these are lists
    # need to be np arrays
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)    
    
# Reshaping
# Can also add other exogenous variables like volume, etc.
# Not being implemented in this model
# Need reshape for np array
# Only using 1 other indicator so third dimension, 1 is number of indicators
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))

# Part -2 Building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialize the RNN
regressor = Sequential()

# Addubg the first LSTM layer and some dropout regularization
# Units is number of neurons 50 
# Units is keras 2.0
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(0.2))

# Adding a second layer with dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


# Part 3 - Making the predictions and visualizing the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values

# Getting the predicted stock price of 2017
# Concatenate original dataframes so testing data is not impacted
# Then just scaling the inputs and not the test values
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
# Since using 60 days need to go back and .values for np array
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)

# Need a 3D structure after scaling them
inputs = sc.transform(inputs)

# Need testing set for predictions
# Predicting 20 financial days in January need 60 + 20 days

X_test = []

for i in range(60, 80):
    # these are lists
    # need to be np arrays
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test) 
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

predicted_stock_price = regressor.predict(X_test)

# Apply the inverse scaling to get the stock price back
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Get the errors
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
rmse_percent = rmse/real_stock_price

# Visualize the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Googl Stock Price')
plt.legend()
plt.show()
