import Tree as T
import Node as n
import Validation as v

root = n.create_root()
T.build_tree(root)
T.show_tree(root,n.tags)

print v.evaluate_test(root,n.x_test)
