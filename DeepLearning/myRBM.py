#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 20:12:43 2018

@author: BryanB
"""

# Boltzmann Machines

import os
os.chdir('/Volumes/Macintosh2TB/Programming/DeepLearningA_Z/BoltzmannMachines')

# import key libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


# import the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# prepare the training set and test set
# there are 5 sets of base = training, test = testing
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')

test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(training_set, dtype = 'int')

# getting the number of users and movies
# use a 0 if the user did not rate a movie
# need to find unique number of users and movies in each set
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# convert the data into an array with users in lines and movies in columns
# rbm requires a certain format
# each user is a item in a list
# contains a list of reviews for all movies
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data    

training_set = convert(training_set)
test_set = convert(test_set)
        
        
        
        
        
        
        
        
        
        
        
        
        
    