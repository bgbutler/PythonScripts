# -*- coding: utf-8 -*-
"""
-----------------------------------------------------------------------------

           Advanced Methods : Breast Cancer

             Copyright : V2 Maestros @2015
                    
Problem Statement
*****************
The dataset contains diagnosis data about breast cancer patients
 and whether they are Benign (healthy) or Malignant
 (possible disease). We need to predict whether new patients 
 are benign or malignant based on model built on this data.

## Techniques Used

1. Principal Component Analysis
2. Training and Testing
3. Confusion Matrix
4. Bagging
5. Boosting


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

os.chdir("C:\Personal\V2Maestros\Modules\Machine Learning Algorithms\Advanced Methods")

"""
Data Engineering and Analysis
"""
#Load the dataset

cancer_data = pd.read_csv("breast_cancer.csv")
cancer_data.dtypes
cancer_data.head()

"""
Principal Component Analysis

In this section, we first scale the data and discover the
 principal components of the data. Then we only pick the 
 top components that have the heaviest influence on the 
 target.
 
"""
from sklearn.decomposition import PCA

predictors = cancer_data.iloc[0:,2:]
targets = cancer_data.diagnosis

#Do PCA
pca=PCA(n_components=4)
reduced_predictors=pca.fit_transform(predictors)
reduced_predictors

#Convert target to integer
targets[targets == 'B']=0
targets[targets == 'M']=1
targets=targets.astype('int64')

#Correlations
DataFrame(reduced_predictors).join(targets).corr()

#Split as training and testing
pred_train, pred_test, tar_train, tar_test  =   train_test_split(DataFrame(reduced_predictors), targets, test_size=.3)

pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape

#Build model on training data

#Using support vector machines
from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier
#classifier=ensemble.BaggingClassifier(DecisionTreeClassifier())
classifier=ensemble.AdaBoostClassifier(DecisionTreeClassifier())


classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

sklearn.metrics.confusion_matrix(tar_test,predictions)
sklearn.metrics.accuracy_score(tar_test, predictions)
sklearn.metrics.classification_report(tar_test, predictions)

