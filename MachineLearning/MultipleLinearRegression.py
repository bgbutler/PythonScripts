# multiple linear regression
# data preprocessing

# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load the data set
import os

dataFile = os.path.normpath("C:/Users/n846490/Documents/Python Scripts/Multiple_Linear_Regression/50_Startups.csv")
dataset = pd.read_csv(dataFile)

# get the X values
X = dataset.iloc[:,0:4].values
X

# create the y value
y = dataset.iloc[:,-1].values
y

# start with making them np.arrays
"""
X = np.array(dataset.iloc[:,0].values)
y = np.array(dataset.iloc[:,1].values)"""

# encoding categorical data
# encoding tends to follow alphabetical order
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X= onehotencoder.fit_transform(X).toarray()

# avoiding the dummy variable trap
X = X[:,1:]


# split the data to training and testing
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

# feature scaling not always used
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting the multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predict the results
y_pred = regressor.predict(X_test)

# visualizing the test set results
plt.scatter(y_pred, y_test, color = 'red')
plt.title('Startup Profits (Test Set)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# building the optimal model using backward elimination
# stats models does not include constant, need to create a 1 coefficent
# need to append the X to the matrix to get the order correct
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

# start with the original matrix ad select the significance level
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# based on output remove index 2
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()



