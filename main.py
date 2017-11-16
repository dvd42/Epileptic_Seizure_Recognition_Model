import Tree as T
import Node as n
import Validation as v

root = n.create_root()
T.build_tree(root)
T.draw_tree(root, n.tags)

accuracy,precision,recall,specificity,f_score = v.evaluate_test(root,n.x_test)
print "Accuracy: %.3f" % accuracy, "Precision: %.3f" % precision, "Recall: %.3f: " % recall,"Specificity: %.3f: " % specificity,"F_score: %.3f" % f_score