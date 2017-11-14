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
    
    #
    # Get elements from each column in with steps of 100
   
    return x_train,x_test,tags

def calculate_mid_points(data):
    
    slices = np.sort(data[:,:-1],axis=0)

    mid_points = slices[slices.shape[0]/2,:]
    
    return mid_points
    
    """
    slices = np.sort(data[:,:-1],axis=0)[0::data.shape[0]*0.065,:]
    
    if slices.shape[0] % 2:
        slices = np.append(slices,slices[-1].reshape((1,2)),axis=0)
    
    
    mid_points = []
    
    # Calculate the mid point for each 2 elements in each column
    for i in range(slices.shape[1]):
        mid_points.append((slices[0::2,i] + slices[1::2,i]) / 2.0)
    
    return mid_points
    """