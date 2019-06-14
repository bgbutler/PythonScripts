# -*- coding: utf-8 -*-
"""
-----------------------------------------------------------------------------

           Linear Regression : Predicting Miles per gallon

                    Copyright : V2 Maestros @2015
                    
Problem Statement
*****************
The input data set contains data about details of various car 
models. Based on the information provided, the goal is to come up 
with a model to predict Miles-per-gallon of a given model.

Techniques Used:

1. Linear Regression ( multi-variate)
2. Data Imputation
3. Variable Reduction
4. Storing Models

-----------------------------------------------------------------------------
"""
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics

os.chdir("C:\Personal\V2Maestros\Modules\Machine Learning Algorithms\Linear Regression")

"""
Data Engineering and Analysis
"""
#Load the dataset
auto_data = pd.read_csv("auto-miles-per-gallon.csv")
auto_data.dtypes
auto_data.describe()
auto_data.head()

"""
The analysis shows that HORSEPOWER is an object, not a int.
This means that there is some text value there in this column
because of which it is not automatically converted to int. This
might be missing data and need to be explored.
"""

#Convert HORSEPOWER to int
for rowNum, colName in auto_data.HORSEPOWER.iteritems():
    if colName.isdigit() == False:
        auto_data.HORSEPOWER[rowNum]= 124
        
auto_data.HORSEPOWER = auto_data.HORSEPOWER.astype(numpy.int64)
auto_data.dtypes

#Find correlations between variables
auto_data.corr()

"""
CYLINDERS, DISPLACEMENT and WEIGHT have very high correlation 
amongst themselves. This is logical in the auto world. Hence
2 of these variables might be proxy of the other. So we will
remove two of these from the dataset
"""
#Remove displacement and cylinders
auto_data.drop('DISPLACEMENT',axis=1, inplace=True)
auto_data.drop('WEIGHT',axis=1, inplace=True)
auto_data.dtypes

#See correlations
plt.scatter(auto_data.MPG, auto_data.ACCELERATION)
plt.cla()
plt.scatter(auto_data.MPG, auto_data.HORSEPOWER)

plt.cla()
plt.boxplot([[auto_data.MPG[auto_data.CYLINDERS==4]],
              [auto_data.MPG[auto_data.CYLINDERS==6]] ,
                [auto_data.MPG[auto_data.CYLINDERS==8]] ],
            labels=('4','6','8'))

"""
Modeling and Prediction
"""
#Split into training and testing sets

predictors = auto_data[['CYLINDERS','HORSEPOWER','ACCELERATION','MODELYEAR']]
targets = auto_data.MPG

pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, targets, test_size=.3)

pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape

#Build model on training data
model = LinearRegression()
model.fit(pred_train,tar_train)

#save and load the model
import pickle
lm_string=pickle.dumps(model)
lm_string

model=pickle.loads(lm_string)

#Test on testing data
predictions = model.predict(pred_test)
predictions

print ( 'R-Squared : ', model.score(pred_test, tar_test))

sklearn.metrics.r2_score(tar_test, predictions)

#Viewing accuracy
plt.cla()
plt.plot(tar_test, tar_test, c='r')
plt.scatter(tar_test, predictions)

#Predicting for a new data point
model.predict([  4. ,  71. ,  16.5,  75. ])

#----------------------------------------------------------------------------
