# -*- coding: utf-8 -*-
"""
@author: Diego
"""

import pandas as pd
import numpy as np
from random import randint
import collections as c
import math as m

import Runtime_Parser as rp


def split_data(data,method):
    """
    :param data: the entire data-set
    :param method: the cross-validation method that will be used
    :return: the train and test sets
    """

    if method == 1:
        x_train = data[:data.shape[0]*2/3:,:]
        x_test = data[x_train.shape[0]:data.shape[0]:,:]


    else:
        index = []
        for i in range(data.shape[0]):
            index.append(randint(0,data.shape[0]-1))

        x_train = data[index]

        x_test = data[np.setdiff1d(np.arange(data.shape[0]),index)]


    return x_train, x_test


def generate_Nan_test(x_test,nans):
    """
    
    :param x_test: the test set
    :param nans: how many you want to insert in the test set
    :return: the test set with nans inserted
    """

    x_test = x_test.astype(float)
    for k in range(nans):
        i = randint(0,x_test.shape[0]-1)
        j = randint(0,x_test.shape[1]-1)
        x_test[i,j] = np.nan

    return x_test


def init_data():

    """
    
    :return: the train and test sets, the mid_points and the names of the columns in the data-set
    """

    # Import the data_set
    data_set = pd.read_csv(rp.data)
    x = data_set.iloc[:,1:].values
    tags = list(data_set)[1:]

    x_train, x_test = split_data(x,rp.cv_method)

    x_test = generate_Nan_test(x_test,rp.nan)

    mid_points = calculate_mid_points(x_train,rp.mid_points)

    return x_train,x_test,mid_points,tags

def calculate_mid_points(data,mid_points):
    """
    :param data: the partial data-set from which to get the mid_points
    :param mid_points: number of desired mid_points
    :return: the new mid points
    """
    data = np.sort(data[:,:-1],axis=0)
    return data[np.mod(np.arange(data.shape[0]),data.shape[0]/mid_points) == 0,:][1:-1]



def entropy_calculation(n,new_index,data):
    """
    
    :param n: the number of samples in the parent node
    :param new_index: index of the data that belongs to the son of current node
    :param data: entire data-set
    :return: the resulting entropy 
    """

    if new_index.shape[0] == 0 or n == 0:
        return 0

    distribution = new_index.shape[0] / float(n)
    p0 = c.Counter(data[new_index][:, -1])[1] / float(new_index.shape[0])
    p1 = 1 - p0

    if p1 == 0 or p0 == 0:
        return 0

    h = sum([m.log(p0,2)*p0,m.log(p1,2)*p1])*-1

    return h * distribution

