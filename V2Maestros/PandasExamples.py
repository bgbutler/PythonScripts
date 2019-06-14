# -*- coding: utf-8 -*-
"""
-----------------------------------------------------------------------------

                       Python - Pandas Examples

                       Copyright : V2 Maestros @2015

-----------------------------------------------------------------------------

 This file contains sample code for demonstrating the capabilities of pandas. Any
 data files required for execution of this code is present with the package.
 Please place all the files in the same folder and set that folder as the current
 working directory.

 It is expected that you have prior programming experience with python language
 Basics of python programming and language contructs are not explained in this
 course.
 
 Comprehensive API reference can be found at
 http://pandas.pydata.org/pandas-docs/dev/api.html 
-----------------------------------------------------------------------------
"""
from pandas import Series, DataFrame
import pandas as pd
import numpy as np

#----------------------------------------------------------------------------
#      Data Structure : Series
#----------------------------------------------------------------------------

#Create a series from an array
mySeries=Series([4,5,2,1])
mySeries

#Assign indexes
mySeries=Series([4,5,2],
                index=['Apples','Oranges','Grapes'])
mySeries

#Data Filtering
mySeries['Oranges']
'Apples' in mySeries

#Add new values
mySeries['Pears']=6
mySeries

#Delete a value
mySeries.drop(['Oranges'])

#Create series from a dictionary
myDict = {'USA':75,'Canada':20}
dictSeries = Series(myDict)
dictSeries

#----------------------------------------------------------------------------
#      Data Structure : Data Frames
#----------------------------------------------------------------------------

#Creating a Data Frame from a Dictionary
empDict = { 'id' : [1,2,3,4],
            'name' : ['Mark','Ian','Sam','Rich'],
            'isManager':[False, True, False, True]}

empDf = DataFrame(empDict)
empDf

#Access rows and columns
empDf['name']
empDf.name
empDf.name[2]
empDf[empDf.isManager == False]
empDf.head()
empDf.tail()
empDf.iloc[2,]

#Create new column
empDf['deptId'] = Series([1,1,2,2])
empDf

#Create new row
empDf.append(Series([5,False,'Derek',2],
                    index=['id','isManager','name','deptId']),
             ignore_index=True)

#Delete a column
empDf['dummy']=1
empDf
del empDf['dummy']
empDf

#Delete a row
empDf.drop(1)

#Sort a Data Frame
empDf.sort_index(axis=1)
empDf.sort(['isManager','name'])

empDf.describe()
empDf.id.corr(empDf.deptId)

#Iterate through a DataFrame
for rowNum, row in auto_data.iterrows():
        for colName, col in row.iteritems():
            #if  pd.isnull(col) :
                print(pd.isnull(col),rowNum, colName)

#----------------------------------------------------------------------------
#                   Data Operations
#----------------------------------------------------------------------------


carDict = { 'ID' : [1,2,3,4,5],
            'MODEL' : ['Taurus','Edge','Camry','Corolla','HighLander'],
            'MAKE' : ['Ford','Ford','Toyota','Toyota','Toyota'],
            'PRICE' : [30,35,27,18,40],
            'WEIGHT' : [2.3,3.5,2.1,1.8,3.7]        
        }
carDf = DataFrame(carDict)
carDf

#Group By
grouped=carDf['PRICE'].groupby(carDf.MAKE)
grouped
grouped.describe()
grouped.mean()
grouped.max()

for make, price in grouped:
        print make
        print price


#----------------------------------------------------------------------------
#      Graphics : matplotlib
# 
#Examples can be found at http://matplotlib.org/examples/index.html
#----------------------------------------------------------------------------
import matplotlib.pylab as plt
import numpy as np


data1 = range(100)
data2 = [ value % 5 + 10 for value in data1]
data3 = np.sin(data1)

#Plot x-y
plt.cla()
plt.plot(data1, data2)
plt.cla()
plt.plot(data1, data3)

#Plot scatter
plt.cla()
plt.scatter(carDf.PRICE, carDf.WEIGHT, color='r')

#Plot bar charts
plt.cla()
plt.bar(carDf.ID, carDf.PRICE)
plt.cla()
plt.barh(carDf.ID, carDf.WEIGHT)
plt.yticks(carDf.ID, carDf.MODEL)

#Plot a pie chart
plt.cla()
plt.pie(carDf.PRICE, labels=carDf.MODEL, 
        shadow=True, autopct='%1.1f')

#Plot a histogram
plt.cla()
plt.hist(data3, color='g')
plt.title("Demo Histogram")
plt.xlabel("Sin weights")
plt.ylabel("Frequency")

#Plot a box plot
plt.cla()
#Pass a List of Lists
plt.boxplot([[carDf.WEIGHT[carDf.MAKE=='Toyota']],
              [carDf.WEIGHT[carDf.MAKE=='Ford']]  ],
            labels=('Toyota','Ford'))


#----------------------------------------------------------------------------
#                   Data Acquisition
#----------------------------------------------------------------------------

import os
os.chdir("C:/Personal/V2Maestros/Modules/Python - Pandas")

#File
irisData = pd.read_csv("iris.csv")
irisData
irisData.describe()
irisData['dummy']=1
irisData.to_csv("iris_extended.csv")

#DB
import mysql.connector
cnx = mysql.connector.connect(user='root', password='',
                              host='localhost',
                              database='demo')
cursor = cnx.cursor()
cursor.execute('select * from demotable')
df = DataFrame(cursor.fetchall())
df.columns = cursor.column_names
df
cnx.close()

#Web
flightDf = pd.read_csv("https://sourceforge.net/p/openflights/code/HEAD/tree/openflights/data/airlines.dat?format=raw")
flightDf

#----------------------------------------------------------------------------
#                   Data Transformation
#----------------------------------------------------------------------------

#Merge Data Frames
empDf

deptDict = { 'id' : [1,2],
            'name' : ['Sales','Engineering']}
deptDf = DataFrame(deptDict)
deptDf

pd.merge(empDf,deptDf,left_on='deptId', right_on='id')

#Binning
carDf

carDf['PRICE_RANGE']=pd.cut(carDf.PRICE, [1,30,60])
carDf

#Create indicator/dummy variables
carDf = carDf.join(pd.get_dummies(carDf.MAKE))
carDf

#Centering and Scaling
carDf['CENT_PRICE']=(carDf.PRICE - numpy.mean(carDf.PRICE))/numpy.std(carDf.PRICE)
carDf['CENT_WGT']=(carDf.WEIGHT - numpy.mean(carDf.WEIGHT))/numpy.std(carDf.WEIGHT)
carDf

#----------------------------------------------------------------------------


