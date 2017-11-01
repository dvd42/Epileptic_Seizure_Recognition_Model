# -*- coding: utf-8 -*-
"""
Created on Wed Nov 01 14:25:18 2017

@author: Diego
"""

import pandas as pd
from scipy import stats as st
import numpy as np
import collections as c


# Import the dataset
dataset = pd.read_csv("data.csv")
x = np.sort(dataset.iloc[:,1:-1].values,axis=0)
y = dataset.iloc[:,-1].values

#TODO split dataset into training and test

# Get elements from each column in with steps of 100
slices = x[0::100,:]

mid_points = []

# Calculate the mid point for each 2 elements in each column
for i in range(slices.shape[0]-1):
    mid_points.append((slices[i,:] + slices[i + 1,:]) / 2)
    
    
# TODO  Calculate the resulting entropy of branching with each mid point

# This is just an example of calculating the entropy when dividing by -36
branch1 = y[dataset.iloc[:,178].values <= -36]
branch2 = y[dataset.iloc[:,178].values > -36]

distribution1 = branch1.size/float(y.size)
distribution2 = branch2.size/float(y.size)

p0 = c.Counter(branch1)[1]/float(branch1.size)
p1 = 1 - p0
p2 =  c.Counter(branch2)[1]/float(branch1.size)
p3 = 1 - p2

h = st.entropy([p1,p0],base=2) * distribution1  + st.entropy([p2,p3],base=2) * distribution2



