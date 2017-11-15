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
mid_points = d.calculate_mid_points(x_train)

print mid_points.size

class Node():
    
    def __init__(self,mid_points_index ,data_index, parent, left, level,column):
        
        self.mid_points = mid_points_index
        self.best_value = 0
        self.column = column
        self.parent = parent
        self.left = left
        self.current_entropy = entropy_calculation(data_index.shape[0],data_index)
        self.level = level
        self.data_index = data_index
        self.leaf = False
        self.son_1 = None
        self.son_2 = None


        print "entropy",self.current_entropy

        if self.current_entropy == 0 or mid_points[self.mid_points].size == 0 or self.data_index.size == 0:
            self.leaf = True

    def branch(self):
        
        print "mid_points",mid_points[self.mid_points].size/178
        print "data",self.data_index.size
        
        if self.leaf:
            self.son_1 = None
            self.son_2 = None
            return
    

        n = self.parent.data_index.shape[0] if self.parent != None else self.data_index.shape[0]

        d_i = self.data_index
        gain = 0
        for i in range(self.mid_points.shape[1]):
            for value in mid_points[self.mid_points[:,i],i]:
                branch_1 = d_i[x_train[d_i, i] < value]
                branch_2 = d_i[x_train[d_i,i] >= value]

                h1 = entropy_calculation(n,branch_1)
                h2 = entropy_calculation(n,branch_2)

                new_gain = self.current_entropy - (h1 + h2)

                if gain < new_gain:
                    gain = new_gain
                    self.best_value = value
                    self.column = i%self.mid_points.shape[1]
                    index_1 = branch_1
                    index_2 = branch_2


        mid_points_index_1  = np.copy(self.mid_points)
        mid_points_index_2 = np.copy(self.mid_points)
        mid_points_index_1[:,self.column] = mid_points[:,self.column] < self.best_value
        mid_points_index_2[:,self.column] = mid_points[:,self.column] >= self.best_value
                            
        
        n1 = Node(mid_points_index_1,index_1,self,True, self.level + 1,self.column)
        n2 = Node(mid_points_index_2,index_2,self,False, self.level + 1,self.column)
    
        self.son_1 = n1
        self.son_2 = n2

def entropy_calculation(n,new_index):
    
    if new_index.shape[0] == 0 or n == 0:
        return 0
    
    distribution = new_index.shape[0] / float(n)
    p0 = c.Counter(x_train[new_index][:, -1])[1] / float(new_index.shape[0])
    p1 = 1 - p0

    if p1 == 0 or p0 == 0:
        return 0       
    
    h = sum([m.log(p0,2)*p0,m.log(p1,2)*p1])*-1


    return h * distribution


def build_tree(node):

    node.branch()

    print node, node.son_1 if not node.leaf else None, node.son_2 if not node.leaf else None

    if node.son_1 != None:
        node = build_tree(node.son_1)

    if node.son_2 != None:
        build_tree(node.son_2)
    
    return node.parent

d_i = np.arange(x_train.shape[0]).T
root = Node(np.ones((mid_points.shape),dtype="bool"),d_i,None,False,0,0)
print "left branch\n"
build_tree(root)




"""
node = root
node2 = root
while (node != None and node2 != None):

    if node != None:
        if not node.leaf:
            print node.level, node.son_1.level, node.son_2.level
            node = node.son_1

    if node2 != None:
        if not node2.leaf:
            print node.level, node.son_1.level, node.son_2.level
        node2 = node.son_2
"""

