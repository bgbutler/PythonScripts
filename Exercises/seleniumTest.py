# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 16:49:19 2016

@author: n846490
"""

# testing the selenium chrome driver

import time
import os
from selenium import webdriver

# set the path to the chromedriver.exe
driver_file = os.path.normpath("C:/Users/n846490/AppData/Local/Continuum/Anaconda3/pkgs/chromedriver.exe")


# do a test opens a google page
# does a search and then closes the page
driver = webdriver.Chrome(driver_file)  # Optional argument, if not specified will search path.
driver.get('http://www.google.com/xhtml');
time.sleep(5) # Let the user actually see something!
search_box = driver.find_element_by_name('q')
search_box.send_keys('ChromeDriver')
search_box.submit()
time.sleep(5) # Let the user actually see something!
driver.quit()