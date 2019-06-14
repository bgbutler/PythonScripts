# Upper Confidence Bound

# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load the data set
import os

# clean everything
# %reset -f

dataFile = os.path.normpath("~/Documents/Programming/TrainingDataResources/UCB/Ads_CTR_Optimisation.csv")
dataset = pd.read_csv(dataFile)

model_type = 'UCB - Random'

# Implementing Random Selection
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward
    
# visualize the results  - histogram
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()







