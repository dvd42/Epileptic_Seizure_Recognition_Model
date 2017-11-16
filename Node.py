import numpy as np
import math as m
import collections as c

import Process_data as d
import Runtime_Parser as rp


x_train,x_test,mid_points,tags = d.init_data()

class Node():

    def __init__(self,mid_points_index ,data_index, parent,level,column):

        """
        :param level: the current tree depth 
        :param column: index of the attribute that best divides the current data
        :self.epilepsy: the class (1,positive !1 negative)
        :self.distribution: the amount of samples belonging to each class
        """

        self.mid_points = mid_points_index
        self.best_value = 0
        self.column = column
        self.parent = parent
        self.current_entropy = d.entropy_calculation(data_index.shape[0],data_index,x_train)
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
        """
        Expands current node 
        """

        if self.check_leaf():
            return

        n = self.parent.data_index.shape[0] if self.parent != None else self.data_index.shape[0]
        index_1, index_2 = self.get_best_attribute(n)
        self.create_sons(index_1, index_2)

    def create_sons(self, index_1, index_2):
        mid_points_index_1 = np.copy(self.mid_points)
        mid_points_index_2 = np.copy(self.mid_points)
        mid_points_index_1[:, self.column] = mid_points[:, self.column] < self.best_value
        mid_points_index_2[:, self.column] = mid_points[:, self.column] >= self.best_value
        n1 = Node(mid_points_index_1, index_1, self, self.level + 1, self.column)
        n2 = Node(mid_points_index_2, index_2, self, self.level + 1, self.column)
        self.son_1 = n1
        self.son_2 = n2

    def get_best_attribute(self, n):
        """
        Finds the attribute that maximizes the gain
        :param n: number samples in current node
        
        """

        gain = 0
        for i in range(self.mid_points.shape[1]):
            for value in mid_points[self.mid_points[:, i], i]:
                branch_1 = self.data_index[x_train[self.data_index, i] < value]
                branch_2 = self.data_index[x_train[self.data_index, i] >= value]

                h1 = d.entropy_calculation(n, branch_1,x_train)
                h2 = d.entropy_calculation(n, branch_2,x_train)

                if rp.alg == 1:
                    split_info = 1

                else:
                    p0 = branch_2.size / self.data_index.size
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


        return index_1, index_2

    def check_leaf(self):
        """     
        :return: True is self is a leaf, False otherwise
        """

        if self.leaf:
            self.distribution = c.Counter(x_train[self.data_index][:, -1])
            if self.distribution[1] / self.data_index.shape[0] == 1:
                self.epilepsy = True

            self.son_1 = None
            self.son_2 = None
            return True

        return False

def create_root():
    root = Node(np.ones((mid_points.shape), dtype="bool"),np.arange(x_train.shape[0]).T,None,0, 0)
    return root
