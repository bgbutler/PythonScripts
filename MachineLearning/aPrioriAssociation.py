# Apriori

# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load the data set
import os

# clean everything
# %reset -f

dataFile = os.path.normpath("~/Documents/Programming/TrainingDataResources/Apriori_Python/Market_Basket_Optimisation.csv")
dataset = pd.read_csv(dataFile, header = None)

model_type = 'Association Learning'

# adjust the working directory for apyiori.py
working_dir = os.path.normpath("/Users/Bryan/Documents/Programming/TrainingDataResources/Apriori_Python")
os.chdir(working_dir)
os.getcwd()

# apriori needs a list of lists
transactions = []
end_transactions = len(dataset)

for i in range(0,end_transactions):
    transactions.append([str(dataset.values[i, j]) for j in range(0,20)])

# Training Apriori on the dataset

from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# visualizing the results
results = list(rules)









    
