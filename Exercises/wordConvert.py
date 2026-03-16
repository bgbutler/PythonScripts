# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

def y_replacer(original):
    replaced = original.replace('y', 'i')
    return replaced


def get_word():
    y_word = raw_input('Input a word with at least one y: ')
    print(y_word)

    converted = y_replacer(y_word)
    print(y_word + ' was replaced with ' + converted)

get_word()