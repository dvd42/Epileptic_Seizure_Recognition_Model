import numpy as np

def test(node, sample):
    """
    
    :param node: current node
    :param sample: sample that is being classified
    :return: it will ultimately return the classification of this sample and the number of positives elements 
    """

    if node.leaf:
        return node.epilepsy,node.distribution[1]

    # Predict class if there is an unknown attribute value in the sample
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


def evaluate_test(root, x_test):
    """
    
    :param root: root node
    :param x_test: numpy array holding the samples we want to classify
    :return: the classification metrics
    """

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    # Classify each sample
    for sample in x_test:
        classification, garbage = test(root, sample)
        if classification and sample[-1] == 1:
            TP += 1
        elif classification and sample[-1] != 1:
            FP += 1
        elif not classification and sample[-1] == 1:
            FN += 1
        else:
            TN += 1

    # Calculate metrics
    accuracy = (TP + TN) / float((TP + TN + FP + FN))
    precision = TP / float(TP + FP)
    recall = TP / float(TP + FN)
    specificity = TN / float((TN + FP))
    f_score = 2 * precision * recall / (precision + recall)

    print "TP: %d" % TP, "FP: %d" % FP, "FN: %d" % FN, "TN: %d" % TN
    return accuracy,precision, recall,specificity,f_score