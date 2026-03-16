# -*- coding: utf-8 -*-

# data preprocessing

# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# load the data set
import os

dataFile = os.path.normpath("C:/Users/n846490/Documents/DataScience/CourseWork/Data.csv")
dataset = pd.read_csv(dataFile)

# get the x values
X = dataset.iloc[:,:-1].values

# can also use print(X)
X

# create the Y value
Y = dataset.iloc[:,3].values
Y

# handling some missing data
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
X

# encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X= onehotencoder.fit_transform(X).toarray()

# now do the dependent variable
labelencoder_Y= LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


# split the data to training and testing
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)









