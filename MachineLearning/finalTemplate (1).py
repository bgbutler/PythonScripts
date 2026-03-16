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
X

# create the Y value
Y = dataset.iloc[:,3].values
Y


# split the data to training and testing
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)


# feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

