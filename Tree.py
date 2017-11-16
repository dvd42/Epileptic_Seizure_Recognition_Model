#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 09:13:02 2017

@author: diego
"""

def build_tree(node):

    node.branch()

    if node.son_1 != None:
        node = build_tree(node.son_1)

    if node.son_2 != None:
        build_tree(node.son_2)

    return node.parent

def show_tree(node,tags):

    if node.son_1 == None and node.son_2 == None:
        return

    print "=="*node.level + " Data : %d Positives: %d Negatives: %d, %s %d "  %(node.data_index.size,node.distribution[1],node.data_index.size - node.distribution[1], tags[node.column],node.best_value)
    show_tree(node.son_1,tags)
    show_tree(node.son_2,tags)




