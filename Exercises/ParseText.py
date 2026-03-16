# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

This script imports text files and turns them into dictionaries
"""

import sys
import os

myPath = os.path.join('/Users', 'Bryan', 'Downloads')


print myPath

os.chdir(myPath)
os.listdir(os.curdir)

myFile = "Deck Safety August 2014.txt"

file_object = open(myFile, 'r')
count = 0

while file_object.readline() !="":
        count = count + 1
print count," lines in the document"

with open(myFile) as myFile:
    head = [next(myFile) for x in xrange(100)]
print head

file_object.close() 




#with open(myFile) as f:
 #   for line in f:
        
    
   