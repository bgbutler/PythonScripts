# decision tree

# data preprocessing

# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load the data set
import os

dataFile = os.path.normpath("C:/Users/n846490/Documents/Python Scripts/Decision_Tree_Regression/Position_Salaries.csv")
dataset = pd.read_csv(dataFile)

# get the X values
X = dataset.iloc[:,1:2].values
X

# create the y value
y = dataset.iloc[:,2].values
y

# feature scaling not always used
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""


# Fitting the multiple linear regression to the training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# making the final prediction for the position 6.5
y_pred = regressor.predict(6.5)

# visualize decision tree results
# make a plot to visualize the training  set results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'green')
plt.title('Truth or Bluff (Decision Tree)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()







