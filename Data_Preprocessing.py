# -*- coding: utf-8 -*-
"""
@author: Diego
"""

import pandas as pd
import numpy as np
from random import randint

def split_data(data,method="holdout"):

    if method == "holdout":
        x_train = data[:data.shape[0]*2/3:,:]
        x_test = data[x_train.shape[0]:data.shape[0]:,:]


    if method == "bootstrapping":
        index = []
        for i in range(data.shape[0]):
            index.append(randint(0,data.shape[0]-1))

        x_train = data[index]

        x_test = data[np.setdiff1d(np.arange(data.shape[0]),index)]


    return x_train, x_test



def generate_Nan_test(x_test,nans=5):

    x_test = x_test.astype(float)
    for k in range(nans):
        i = randint(0,x_test.shape[0]-1)
        j = randint(0,x_test.shape[1]-1)
        x_test[i,j] = np.nan

    return x_test


def init_data():

    # Import the dataset
    dataset = pd.read_csv("data.csv")
    x = dataset.iloc[:,1:].values
    tags = list(dataset)[1:]

    x_train, x_test = split_data(x,"holdout")

    return x_train,x_test ,tags

def calculate_mid_points(data,mid_points=10):
    data = np.sort(data[:,:-1],axis=0)
    a = data.shape[0]
    b  = a/mid_points
    return data[np.mod(np.arange(data.shape[0]),b) == 0,:][1:-1]



