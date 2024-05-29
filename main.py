from BinaryTree import BinaryTree
from GiniImpurity import GiniImpurity
import pandas as pd

if __name__ == "__main__":
    tree = BinaryTree()
    data = pd.read_csv(
        "/home/philipp/Documents/semester_6/maschinelles_lernen/uebungen/uebung_3/iris.data",
        header=None)
    gini = GiniImpurity(data, 1)
    gini.split_data_along_cutting_point()
    data = gini.data_chunks
    tree.fill_tree(data)

    tree.print_tree(tree.root)
