# this file loads in text files

import os
import glob2
import datetime

my_path = '/Users/Bryan/Documents/Programming/Udemy_Python/Sample-Files'

os.chdir(my_path)


filenames = glob2.glob('*.txt')


def combine_file_data(files):
    with open(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f') + '.txt', 'w') as file:
        for filename in files:
            with open(filename, 'r') as f:
                file.write(f.read()+'\n')
            
            


combine_file_data(filenames)