# -*- coding: utf-8 -*-
"""
-----------------------------------------------------------------------------

           Association Rules Mining : Accidents Analysis

             Copyright : V2 Maestros @2015
                    
Problem Statement
*****************
The input dataset contains information about 1000 fatal 
accidents. It has different feature variables associated 
with the accident. The goal is to find patterns in the 
variables - which accident conditions frequently occur 
together.

## Techniques Used

1. Association Rules Mining
2. Converting Feature data into Basket Data


-----------------------------------------------------------------------------
"""

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
import apriori

os.chdir("C:\Personal\V2Maestros\Modules\Machine Learning Algorithms\Association Rules  Mining")

"""
Data Engineering and Analysis
"""
#Load the dataset
accident_data = pd.read_csv("accidents.csv")
accident_data.dtypes
accident_data.describe()
accident_data.head()

"""
The data needs to be in basket form. So we are going to convert
the table form into a basket form manually
"""

basket_str = ""
for rowNum, row in accident_data.iterrows():
    
    #Break lines
    if (rowNum != 0):
        basket_str = basket_str + "\n"
    #Add the rowid as the first column
    basket_str = basket_str + str(rowNum) 
    #Add columns
    for colName, col in row.iteritems():
           if ( colName != 'Accident_Index'):
               basket_str = basket_str + "," + colName + "=" + str(col)

print basket_str
basket_file=open("accident_basket.csv","w")
basket_file.write(basket_str)
basket_file.close()

"""
Read the basket file now and compute rules
"""
import csv
with open('accident_basket.csv', 'rb') as f:
    reader = csv.reader(f)
    your_list = list(reader)

 
L,supportData=apriori.apriori(your_list,0.6)
brl=apriori.generateRules(L, supportData,0.6)

for row in brl:
    print list(row[0]), " => ", list(row[1]), row[2]   
    