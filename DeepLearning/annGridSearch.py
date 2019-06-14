#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 13:35:14 2018

@author: Bryan
"""

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# one labelencoder for the country
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

# other encoder for gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# drop the first column to avoid dummy variable trap
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense


# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# for the grid search
from sklearn.model_selection import GridSearchCV

# Initialising the ANN
# For each hyperparameter to tune, needs to be an argument

def build_classifier(optimizer):
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    # there are 11 inputs and the layer will contain 6 nodes
    # add dropout to first layer
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    
    
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 6))
    
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid', input_dim = 6))
    
    # Compiling the ANN
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

# Grid search to tune parameters
# Powers of 2 for batch size
parameters = {'batch_size': [25, 32],
              'nb_epoch': [100, 500],
              'optimizer': ['adam', 'rmsprop']
                }

grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)


grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_












