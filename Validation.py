import numpy as np

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


def evaluate_test(root, x_test):
    TP = 0
    FP = 0
    FN = 0
    TN = 0

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

    accuracy = (TP + TN) / float((TP + TN + FP + FN))
    precision = TP / float(TP + FP)
    recall = TP / float(TP + FN)
    specificity = TN / float((TN + FP))
    f_score = 2 * precision * recall / (precision + recall)

    return accuracy, precision, recall, specificity, f_score