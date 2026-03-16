# this is the beginning of the cg file
# it has been upgraded to use python 3.4

import os
import re

myPath = os.path.join('/Users', 'Bryan', 'Documents', 'Programming', 'SeaHome')


print (myPath)

os.chdir(myPath)
os.listdir(os.curdir)

myFile = "eng1UTF.txt"

file_object = open(myFile, 'r')
count = 0

while file_object.readline() !="":
        count = count + 1
print (count," lines in the document")

file_object.close()

#this gets the lines separated by questions
#this finds all of the occurences
# best regex so far is (r'\(3\..*?\)'
#with open(myFile, 'r') as myFile:
#	data = myFile.read()
#	pattern = re.compile(r'\(3\..*?\)', re.DOTALL)
#	print(pattern.findall(data))
#
#	print("Showing the first 10 items")
#	
#	questions = []
#	questions = pattern.findall(data)[0:10]
#	for i in range (0,10):
#		print(questions[i])
