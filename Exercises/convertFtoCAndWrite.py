
# convert fahrenheit to celsius

import os

my_path = '/Users/Bryan/Documents/Programming/Udemy_Python/'

os.chdir(my_path)

temperatures = [10, -20, -289, 100]


def c_to_f(c):

    f = c*9/5 + 32

    return f


for temp in temperatures:
    if temp > -273.3:
        print(c_to_f(temp))
        with open('new_temperatures.txt', 'a+') as temps:
            temps.write(str(c_to_f(temp)) + '\n')




