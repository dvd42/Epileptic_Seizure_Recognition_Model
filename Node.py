#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 09:13:02 2017

@author: diego
"""

from scipy import stats as st
import collections as c
import numpy as np
import math as m

import Data_Preprocessing as d

x_train,x_test, mid_points,tags = d.init_data()


class Node():
    
    def __init__(self, mid_points, data_index, parent, current_entropy, tag, level):
        self.mid_points = mid_points
        self.parent = parent
        self.current_entropy = current_entropy
        self.tag = tag
        self.level = level
        self.data_index = data_index
        self.leaf = False

        if self.current_entropy == 0:
            self.leaf = True

    def branch(self):


        if self.leaf:
            return None,None

        if self.parent != None:
            n = self.parent.data_index.shape[0]
        else:
            n = self.data_index.shape[0]
                
        d_i = self.data_index
        gain = 0
        for i in range(len(mid_points)):
            for value in mid_points[i]:
                branch_1 = d_i[x_train[d_i, i] < value]
                branch_2 = d_i[x_train[d_i,i] >= value]
                
                
                h1 = entropy_calculation(n,branch_1)
                h2 = entropy_calculation(n,branch_2)

                new_gain = self.current_entropy - (h1 + h2)
    
                if gain < new_gain:
                    gain = new_gain
                    print self.level
                    best_h1 = h1
                    best_h2 = h2
                    best_value = value
                    index_1 = branch_1
                    index_2 = branch_2
                    column = i
                    tag1 = tags[i] + " < " + str(value)
                    tag2 = tags[i] + " >= " + str(value)

        sample_1 = self.mid_points[:]
        sample_2 = self.mid_points[:]
        sample_1[column] = sample_1[column][sample_1[column] < best_value]
        sample_2[column] = sample_2[column][sample_2[column] > best_value]

        #if all(v.size == 0 for v in sample_1) or index_1.size == 0 or self.current_entropy == 0:
            #self.leaf = True


        n1 = Node(sample_1,index_1,self,best_h1,tag1, self.level + 1)
        n2 = Node(sample_2,index_2,self,best_h2,tag2, self.level + 1)
        
        return n1,n2

def entropy_calculation(n,new_index):

    distribution = x_train[new_index].shape[0] / float(n)
    p0 = c.Counter(x_train[new_index][:, -1])[1] / float(x_train[new_index].shape[0])
    p1 = 1 - p0
    
    if p1 == 0 or p0 == 0:
        return 0       
    
    h = sum([m.log(p0,2)*p0,m.log(p1,2)*p1])*-1


    return h * distribution


def build_tree(n1,n2, tree):

    if n1 == None and n2 == None:
        return tree
    
    n1,n3 = n1.branch()
    n2,n4 = n2.branch() 
    
    tree.append(build_tree(n1,n3,tree),build_tree(n2,n4,tree))
    
    return tree

d_i = np.arange(x_train.shape[0])
h = entropy_calculation(d_i.shape[0],d_i)
root = Node(mid_points,d_i,None,h,"root",0)
tree = [root]

n1,n2 = root.branch()

print build_tree(n1,n2,tree)

