# -*- coding: utf-8 -*-
"""
@author: Diego
"""

import pandas as pd
import numpy as np

def init_data():

    # Import the dataset
    dataset = pd.read_csv("data.csv")
    x = dataset.iloc[:,1:].values
    tags = list(dataset)

    # Split dataset into training and test

    # TODO implement other cross_validation methods
    # Using holdout
    x_train = x[:x.shape[0]*2/3:,-3:]
    x_test = x[x_train.shape[0]:x.shape[0]:,:]

    # Get elements from each column in with steps of 100
    slices = np.sort(x_train[:,:-1],axis=0)[0::505,:]

    # TODO fix uneven slices shape 
    
    mid_points = []

    # Calculate the mid point for each 2 elements in each column
    for i in range(slices.shape[1]):
        mid_points.append((slices[0::2,i] + slices[1::2,i]) / 2.0)

    return x_train,x_test,mid_points,tags

