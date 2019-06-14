#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 20:13:39 2017

@author: Bryan
"""

import time
import datetime

lst = []

for i in range(5):
    lst.append(datetime.datetime.now())
    time.sleep(1)
    
print(lst)  