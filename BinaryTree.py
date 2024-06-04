import pandas as pd


class CARTNode:
    NODE_COUNT = 0

    def __init__(self, cutting_point, split_point, previous_attribute, attribute, gini, data, parent=None):
        CARTNode.NODE_COUNT += 1
        self.Id = self.NODE_COUNT
        self.cutting_point = cutting_point
        self.split_point = split_point
        self.cutting_axis = attribute
        self.previous_cutting_axis = previous_attribute
        self.data = data
        self.dominant_class = self.get_dominant_class()
        self.parent = parent
        self.gini_value = gini
        self.left = None
        self.right = None

    def get_dominant_class(self):
        class_column = "class" if "class" in self.data.columns else self.data.columns[-1]
        class_counts = self.data[class_column].value_counts()
        return class_counts.idxmax()

    def get_node_type(self):
        if not self.parent:
            return "Root"
        elif self.left and self.right:
            return "Branch"
        else:
            return "Leaf"

    def __repr__(self):
        parent_id = self.parent.Id if self.parent else "None"
        parent_len = len(self.parent.data) if self.parent else len(self.data)
        return (f"Node: {self.Id}, Parent_Node: {parent_id}. Split_point: {self.cutting_point}, "
                f"Split_column: {self.cutting_axis}, Previous_split_point: {self.split_point}, "
                f"Previous_split_column: {self.previous_cutting_axis}, "
                f", Gini: {self.gini_value}, Size: {len(self.data)}/{parent_len}")


class BinaryTree:
    def __init__(self):
        self.root = None

    def add(self, node):
        """
        Adds a root node if no root node exists. Otherwise, it calls the _add() function to add a node at the correct
        position.
        @param node: The node to be added.
        @return: None
        """
        if node.parent is None:
            self.root = node
        else:
            parent_node = self.find(node.parent.Id)
            split_point = node.data.iloc[:, parent_node.cutting_axis].max()

            if split_point <= parent_node.cutting_point:
                if parent_node.left is not None:
                    parent_node.left = node
                    print(f"Node {node.Id} added left to node {node.parent.Id} (Cutting_point: "
                          f"{parent_node.cutting_point}, split_point {split_point})")
            else:
                parent_node.right = node
                print(f"Node {node.Id} added right to node {node.parent.Id} (Cutting_point: "
                      f"{parent_node.cutting_point}, split_point {split_point})")

    def find(self, node_id):
        return self._find(node_id, self.root)

    def _find(self, node_id, node):
        """
        Iterates through the binary tree and compares the desired key with the key of the current node. If the
        keys are equal, the desired node has been found. Otherwise, it recursively searches in the left and right
        branches until the correct node is found.
        @param node_id: The ID of the node to search for.
        @param node: The current node being examined.
        @return: The node with the corresponding ID, or None if not found.
        """
        if node is None:
            return None
        elif node_id == node.Id:
            return node
        else:
            left_result = self._find(node_id, node.left)
            if left_result:
                return left_result
            return self._find(node_id, node.right)

    def fill_tree(self, data):
        for node in data:
            self.add(node)

    def eval(self, data: pd.DataFrame) -> str:
        """
        Evaluates a new data point and determines the predicted class
        @param data: Dataframe containing the new datapoint to be evaluated
        @return: A string signifying the label of the dominant class
        """
        current_node = self.root
        while current_node.left and current_node.right:
            eval_attribute = current_node.cutting_axis
            eval_value = current_node.cutting_point
            if data.iloc[0, eval_attribute] <= eval_value:
                current_node = current_node.left
            else:
                current_node = current_node.right
        print(current_node.dominant_class)
        return current_node.dominant_class

    def print_tree(self, node, level=0):
        if node is not None:
            self.print_tree(node.right, level + 1)
            parent_size = len(node.parent.data) if node.parent else len(node.data)
            print(
                ' ' * 4 * level + '-> ' + f'Node: {node.Id}, {node.get_node_type()}, {node.dominant_class} '
                                          f'Split_point: {node.cutting_point}, '
                                          f'Split_column: {node.cutting_axis}, '
                                          f'Parent_Node: {node.parent.Id if node.parent else "Root"}, '
                                          f'Size: {len(node.data)}/{parent_size}')
            self.print_tree(node.left, level + 1)
