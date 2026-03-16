# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 19:11:47 2016

@author: Bryan
"""

emails = ['me@gmail.com', 'you@hotmail.com', 'they@gmail.com']

def check_gmail(email_list):    
        for email in email_list:
            if email.find('gmail') != -1:
                print('Found ' + email)
            
check_gmail(emails)