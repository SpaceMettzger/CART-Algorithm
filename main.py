from BinaryTree import BinaryTree
from GiniImpurity import GiniImpurity
import pandas as pd

if __name__ == "__main__":
    tree = BinaryTree()
    data = pd.read_csv(
        "/home/philipp/Documents/semester_6/maschinelles_lernen/uebungen/uebung_3/iris.data",
        header=None)

    if False:
        data = data.drop(columns=[0, 1])
        data.columns = range(len(data.columns))

    gini = GiniImpurity(data, 3, 0.05)
    gini.split_data_along_cutting_point()
    data = gini.data_chunks
    tree.fill_tree(data)

    print("\nAusgabe Baum:")
    print("=============================================")
    tree.print_tree(tree.root)
    print("=============================================")

    tree.eval(pd.DataFrame([[0, 0, 0, 0]]))
    tree.eval(pd.DataFrame([[1, 3, 3, 1]]))
    tree.eval(pd.DataFrame([[6, 6, 6, 6]]))
