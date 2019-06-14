# data preprocessing

# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load the data set
import os

dataFile = os.path.normpath("C:/Users/n846490/Documents/Python Scripts/Simple_Linear_Regression/Salary_Data.csv")
dataset = pd.read_csv(dataFile)

# get the x values
X = dataset.iloc[:,0].values
X

# create the Y value
y = dataset.iloc[:,1].values
y

# start with making them np.arrays
X = np.array(dataset.iloc[:,0].values)
y = np.array(dataset.iloc[:,1].values)




# split the data to training and testing
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 1/3, random_state = 0)

# convert the features into np arrays
X_train = np.array(X_train).reshape((len(X_train), 1))
X_test = np.array(X_test).reshape((len(X_test), 1))
y_test = np.array(y_test).reshape((len(y_test), 1))
y_train = np.array(y_train).reshape((len(y_train), 1))

# feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# predicting the Test set results
y_pred = regressor.predict(X_test)

# make a plot to visualize the training  set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# visualizing the test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()








