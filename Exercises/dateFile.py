# test file for dates

import datetime

filename = datetime.datetime.now()

# create an empty file

def create_file():
    """ this function creates a new empty file"""
    with open(filename.strftime('%Y-%m-%d-%H-%M') + '.txt','w') as file:
        file.write("test data the time of creation is " + str(filename))
        
        
create_file()
   