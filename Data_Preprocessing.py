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
    tags = list(dataset)[1:]

    # Split dataset into training and test

    # TODO implement other cross_validation methods
    # Using holdout
    x_train = x[:x.shape[0]*2/3:,:]
    x_test = x[x_train.shape[0]:x.shape[0]:,:]


    return x_train,x_test,tags

def calculate_mid_points(data,slices=1000):
    data = np.sort(data[:,:-1],axis=0)
    return data[np.mod(np.arange(data.shape[0]),slices) == 0,:][1:-1]
    
