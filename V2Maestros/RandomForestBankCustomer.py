# -*- coding: utf-8 -*-
"""
-----------------------------------------------------------------------------

           Random Forests : Classifying Bank Customers

             Copyright : V2 Maestros @2015
                    
Problem Statement
*****************
The input data contains surveyed information about potential 
customers for a bank. The goal is to build a model that would 
predict if the prospect would become a customer of a bank, 
if contacted by a marketing exercise.

## Techniques Used

1. Random Forests
2. Training and Testing
3. Confusion Matrix
4. Indicator Variables
5. Binning
6. Variable Reduction

-----------------------------------------------------------------------------
"""
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics

os.chdir("C:\Personal\V2Maestros\Modules\Machine Learning Algorithms\Random Forests")

"""
Data Engineering and Analysis
"""
#Load the dataset

bank_data = pd.read_csv("bank.csv", sep=";")
bank_data.dtypes
bank_data.describe()
bank_data.head()

"""
Data Transformations

Let us do the following transformations

1. Convert age into a binned range.
2. Convert marital status into indicator variables. 
We could do the same for all other factors too, but we 
choose not to. Indicator variables may or may not improve 
predictions. It is based on the specific data set and need to
 be figured out by trials.
"""
bank_data['age'] = pd.cut(bank_data.age,[1,20,40,60,80,100])
bank_data = bank_data.join(pd.get_dummies(bank_data.marital))
del bank_data['marital']
bank_data.head()


#Convert all strings to equivalent numeric representations
#to do correlation analysis
colidx=0
colNames=list(bank_data.columns.values)
for colType in bank_data.dtypes:
    if colType == 'object':
        bank_data[colNames[colidx]]=pd.Categorical.from_array(bank_data[colNames[colidx]]).labels
    colidx= colidx+1
    
bank_data.dtypes
bank_data.describe()

#Find correlations
bank_data.corr()

"""
Based on the correlation co-efficients, let us eliminate default,
 balance, day, month, campaign, poutcome because of very low
 correlation. There are others too with very low correlation,
 but let us keep it for example sake.
"""
del bank_data['default']
del bank_data['balance']
del bank_data['day']
del bank_data['month']
del bank_data['campaign']
del bank_data['poutcome']
bank_data.dtypes
"""
Modeling and Prediction
"""
#Split into training and testing sets

predictors = bank_data[['age','job','education','housing','loan','contact','duration','pdays','previous','divorced','married','single']]
targets = bank_data.y

pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, targets, test_size=.3)

pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape

#Build model on training data
from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(n_estimators=25)
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

sklearn.metrics.confusion_matrix(tar_test,predictions)
sklearn.metrics.accuracy_score(tar_test, predictions)
sklearn.metrics.classification_report(tar_test, predictions)

"""
Impact of tree size on predictio accuracy

Let us try to build different number of trees and see the effect
 of that on the accuracy of the prediction
"""

trees=range(25)
accuracy=np.zeros(25)

for idx in range(len(trees)):
   classifier=RandomForestClassifier(n_estimators=idx + 1)
   classifier=classifier.fit(pred_train,tar_train)
   predictions=classifier.predict(pred_test)
   accuracy[idx]=sklearn.metrics.accuracy_score(tar_test, predictions)
   
plt.cla()
plt.plot(trees, accuracy)

