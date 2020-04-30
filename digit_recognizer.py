# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 11:20:39 2020

@author: drran
"""
# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline

# Importing the dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")