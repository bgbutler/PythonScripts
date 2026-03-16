# artificial neural networks


# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# load the data set
import os

dataFile = os.path.normpath("~/Documents/Programming/TrainingDataResources/Artificial_Neural_Networks/Churn_Modelling.csv")
dataset = pd.read_csv(dataFile)

# get the X values
X = dataset.iloc[:,3:13].values
X
# create the y value
y = dataset.iloc[:,13].values
y

# Encode the categorical features 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# code the countries
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])

# code gender
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# remove one dummy country to avoid the dummy trap
X = X[:,1:]


# split the data to training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = 0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Part 2 - Create the ANN
# import the Keras libraries and packages

import keras

# for initialization of  ann
from keras.models import Sequential

# for the layers 
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)





# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# check the results with a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)









