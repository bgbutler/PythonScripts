# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 15:15:34 2016

@author: n846490
"""

# extract tar files

import tarfile

# set the working directory

import os

tarFiles = os.path.normpath("C:/Users/n846490/AppData/Local/Continuum/Anaconda3/pkgs")
os.chdir(tarFiles)
os.getcwd()

tar = tarfile.open("selenium-3.0.2.tar.gz")
tar.extractall()
tar.close()



