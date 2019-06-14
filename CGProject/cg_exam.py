# this is the beginning of the cg file

import sys
import os

myPath = os.path.join('/Users', 'Bryan', 'Documents', 'Programming', 'SeaHome')


print myPath

os.chdir(myPath)
os.listdir(os.curdir)

myFile = "Deck Safety August 2014.txt"

file_object = open(myFile, 'r')
count = 0

while file_object.readline() !="":
        count = count + 1
print (count," lines in the document")

with open(myFile) as myFile:
	head = [next(myFile) for x in xrange(20)]

file_object.close() 

for i in range(0,20):
	print (head[i])

