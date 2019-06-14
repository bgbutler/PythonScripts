#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 09:07:18 2018

@author: Bryan
"""

# churn modeling with lightgbm
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# load the data set
import os

dataFile = os.path.normpath("~/Documents/Programming/TrainingDataResources/Artificial_Neural_Networks/Churn_Modelling.csv")
dataset = pd.read_csv(dataFile)

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

import lightgbm as lgb
training_data = lgb.Dataset(data = X_train, label = y_train)
params = {'num_levels':31, 'num_trees':100, 'objective':'binary'}
params['metric'] = ['auc', 'binary_logloss']
classifier = lgb.train(params = params,
                       train_set = training_data,
                       num_boost_round = 10)


# for catboost
from catboost import CatBoostClassifier
# classifier  = CatBoostClassifier(iterations = 2,
#                                  depth = 2,
#                                  learning_rate = 1,
#                                  loss_function = 'LogLoss'
#                                  logging_level = 'Verbose')
# classifier.fit(X_train, y_train, cat_features = [0,2,5])
# pred = classifier.predict(X_test)
# prob_pred = classifier.predict_prob(X_test)






prob_pred = classifier.predict(X_test)
y_pred = np.zeros(len(prob_pred))
for i in range(0, len(prob_pred)):
    if prob_pred[i] >= 0.5:
        y_pred[i] = 1
    else:
        y_pred[i] = 0

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
accuracies.mean()
accuracies.std()












