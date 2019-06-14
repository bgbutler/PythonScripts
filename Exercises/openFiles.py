# this is for opening the text file

import os

my_dir_name = os.path.normpath('/Users/Bryan/Documents/Programming/Udemy_Python/example.txt')


list_content = []

# with open(my_dir_name, 'r') as my_file:
#    list_content = my_file.read()
#    my_file.seek(0)
#    print(list_content)

list_content = open(my_dir_name, 'r')
list_content.seek(0)
print(list_content)
list_content.close()

