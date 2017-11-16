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

use_nan = True
algorithm = "C4.5"

x_train,x_test,tags = d.init_data()
mid_points = d.calculate_mid_points(x_train,mid_points=3)

if use_nan:
    x_test = d.generate_Nan_test(x_test)


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
        self.epilepsy = False
        self.distribution = c.Counter(x_train[self.data_index][:,-1])

        if self.current_entropy == 0 or mid_points[self.mid_points].size == 0 or self.data_index.size == 0:
            self.leaf = True

    def branch(self):

        if self.leaf:
            self.distribution = c.Counter(x_train[self.data_index][:, -1])
            if  self.distribution[1] / self.data_index.shape[0]  == 1:
                self.epilepsy = True

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

                if algorithm == "ID3":
                    split_info = 1

                elif algorithm == "C4.5":
                    p0 = branch_2.size/self.data_index.size
                    p1 = 1 - p0
                    if p1 == 0 or p0 == 0:
                        split_info = 1
                    else:
                        split_info = sum([m.log(p0, 2) * p0, m.log(p1, 2) * p1]) * -1


                new_gain = (self.current_entropy - (h1 + h2)) / float(split_info)

                if gain < new_gain:
                    gain = new_gain
                    self.best_value = value
                    self.column = i
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

    if node.son_1 != None:
        node = build_tree(node.son_1)

    if node.son_2 != None:
        build_tree(node.son_2)

    return node.parent

def test(node, sample):

    if node.leaf:
        return node.epilepsy,node.distribution[1]

    if np.isnan(sample[node.column]):
        epilepsy_1, distribution_1 = test(node.son_1,sample)
        epilepsy_2, distribution_2 = test(node.son_2, sample)

        p1 = (distribution_1 + distribution_2)/float(node.data_index.size)

        if p1 >= 0.5:
            return True,distribution_1
        else:
            return False,distribution_2

    if sample[node.column] < node.best_value:
        return test(node.son_1, sample)
    else:
        return test(node.son_2,sample)


def show_tree(node):

    if node.son_1 == None and node.son_2 == None:
        return

    print str(node.level) +  "=="*node.level + " Data : %d Positives: %d Negatives: %d, %s %d "  %(node.data_index.size,node.distribution[1],node.data_index.size - node.distribution[1], tags[node.column],node.best_value)
    show_tree(node.son_1)
    show_tree(node.son_2)

def evaluate_test(root, x_test):

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    for sample in x_test:
        classification,garbage = test(root,sample)
        if classification and sample[-1] == 1:
            TP += 1
        elif classification and sample[-1] != 1:
            FP +=1
        elif not classification and sample[-1] == 1:
            FN += 1
        else:
            TN +=1

    accuracy = (TP+TN)/float((TP + TN + FP +FN))
    precision = TP/float(TP + FP)
    recall = TP/float(TP + FN)
    specificity = TN/float((TN + FP))
    f_score = 2 * precision * recall /(precision + recall)

    return accuracy,precision,recall,specificity,f_score


d_i = np.arange(x_train.shape[0]).T
root = Node(np.ones((mid_points.shape),dtype="bool"),d_i,None,False,0,0)
build_tree(root)
#show_tree(root)


print evaluate_test(root,x_test)

