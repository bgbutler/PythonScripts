# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 18:42:02 2016

@author: Bryan"""


# testing conditionals
def test_value(v):
    if v < 3:
        test_string = 'Less'
    elif v == 3:
        test_string  = 'Equals'
    else:
        test_string = 'Greater'
    return test_string
        
        
def get_value():
    value  = raw_input('Input a number: ')
    print(value)

    value = float(value)
    condition = test_value(value)
    print(str(value) + ' is ' + condition + ' 3')
    
get_value()