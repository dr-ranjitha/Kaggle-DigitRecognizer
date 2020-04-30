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

#Get some idea about the training data
train.head()
train.info()
train.shape
train.describe()
train.describe(include=['O'])#describe(include = ['O']) will show the descriptive statistics of object data types.
train.isnull().sum()
train.tail()
#Get some idea about the testing data
test.head()
test.info()
test.shape
test.describe()
test.describe(include=['O'])#describe(include = ['O']) will show the descriptive statistics of object data types.
test.isnull().sum() #missing values