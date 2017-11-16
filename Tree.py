#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 09:13:02 2017

@author: diego
"""

def build_tree(node):
    """
    :param node: the current node to be expanded
    :return: its a recursive function, it will ultimately return None (which is the parent of the root node)
    """

    node.branch()

    if node.son_1 != None:
        node = build_tree(node.son_1)

    if node.son_2 != None:
        build_tree(node.son_2)

    return node.parent

def draw_tree(node, tags):
    """
    :param node: node to be drawn 
    :param tags: the name of the columns in the original dataset
    :return: prints the tree
    """

    if node.son_1 == None and node.son_2 == None:
        return

    print "=="*node.level + " Data : %d Positives: %d Negatives: %d, %s %d "  %(node.data_index.size,node.distribution[1],node.data_index.size - node.distribution[1], tags[node.column],node.best_value)
    draw_tree(node.son_1, tags)
    draw_tree(node.son_2, tags)




