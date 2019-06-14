# polynomial regression

# data preprocessing

# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load the data set
import os

dataFile = os.path.normpath("C:/Users/n846490/Documents/Python Scripts/Polynomial_Regression/Position_Salaries.csv")
dataset = pd.read_csv(dataFile)

# get the X values
X = dataset.iloc[:,1:2].values
X

# create the y value
y = dataset.iloc[:,2].values
y

# data is too small for training and testing
# will compare a linear to a polynomial
# fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#fits a polynomial regress, degree = highest power
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# visualize linear regression results
# make a plot to visualize the training  set results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'green')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# make a plot to visualize the training  set results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'green')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# making the final prediction for the position 6.5
lin_reg.predict(6.5)

lin_reg_2.predict(poly_reg.fit_transform(6.5))

