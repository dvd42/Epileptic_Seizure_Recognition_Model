#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 09:13:02 2017

@author: diego
"""

import collections as c
import numpy as np
import math as m

import Data_Preprocessing as d

x_train,x_test,tags = d.init_data()

class Node():
    
    def __init__(self, data_index, parent, left, level):
        self.mid_points = d.calculate_mid_points(x_train[data_index])
        self.best_value = 0
        self.column = 0
        self.parent = parent
        self.left = left
        self.current_entropy = entropy_calculation(data_index.shape[0],data_index)
        self.level = level
        self.data_index = data_index
        self.leaf = False
        self.son_1 = None
        self.son_2 = None


        if self.current_entropy == 0:
            self.leaf = True

    def branch(self):
        
        if self.leaf:
            return None,None

        n = self.data_index.shape[0]
        d_i = self.data_index
        gain = 0
        for i in range(len(self.mid_points)):
            for value in self.mid_points[i]:
                branch_1 = d_i[x_train[d_i, i] < value]
                branch_2 = d_i[x_train[d_i,i] >= value]
                
                h1 = entropy_calculation(n,branch_1)
                h2 = entropy_calculation(n,branch_2)
    
                new_gain = self.current_entropy - (h1 + h2)
                
                if gain < new_gain:
                    gain = new_gain
                    self.best_value = value
                    self.column = i
                    index_1 = branch_1
                    index_2 = branch_2
                
        if self.parent != None:    
            print tags[self.column] + " < " if self.left else " >= " + str(self.best_value)
        else:
            print "root"
        
        n1 = Node(index_1,self,True, self.level + 1)
        n2 = Node(index_2,self,False, self.level + 1)
    
        self.son_1 = n1
        self.son_2 = n2
    
        return n1,n2

def entropy_calculation(n,new_index):

    distribution = new_index.shape[0] / float(n)
    p0 = c.Counter(x_train[new_index][:, -1])[1] / float(new_index.shape[0])
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
    
    tree.append(build_tree(n1,n3,tree))
    tree.append(build_tree(n2,n4,tree))
    
    return tree

d_i = np.arange(x_train.shape[0]).T
root = Node(d_i,None,False,0)
tree = [root]

n1,n2 = root.branch()

print build_tree(n1,n2,tree)

