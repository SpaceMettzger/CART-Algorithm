class Node:
    def __init__(self, value, parent=None):
        self.value = value
        self.parent = parent
        self.left = None
        self.right = None

    def __repr__(self):
        return f"Node: {self.value}. Parent: {self.parent}, children: {self.left}, {self.right}"


class BinaryTree:
    def __init__(self):
        self.root = None

    def add(self, value):
        """
        Adds a root node if no rood node exists. Else it calls the _add() function to add a node at the correct position
        @param value: the key of the node to be added.
        @return: None
        """
        if not self.root:
            self.root = Node(value)
        else:
            self._add(value, self.root)

    def _add(self, value, node: Node):
        """
        Adds a node at the correct position. If the key is smaller than the parent node key, the node is added left.
        Otherwise, it is added to the right. If a left or right node already exists, the new node is added as a subnode
        to the already existing node.
        @param value: the key of the node to be added.
        @param node: the parent node.
        @return: None
        """
        if value < node.value:
            if node.left is None:
                node.left = Node(value, node)
            else:
                self._add(value, node.left)
        else:
            if node.right is None:
                node.right = Node(value, node)
            else:
                self._add(value, node.right)

    def find(self, value):
        return self._find(value, self.root)

    def _find(self, value, node):
        """
        Iterates through the binary tree and compares the desired key with the key of the current node. If the
        keys are equal, the desired node has been found. If the key is smaller than the key of the node, the desired
        node is in the left branch of the current node. If the key is larger, it is in the right branch. In those cases,
        the node in the desired branch is searched next until the correct node has been found.
        @param value:
        @param node:
        @return:
        """
        if node is None:
            return None
        elif value == node.value:
            return node
        elif value < node.value:
            return self._find(value, node.left)
        else:
            return self._find(value, node.right)

    def fill_tree(self, data):
        self.root =Node(sum([len(value['data']) for value in data.values()]))
        for key in data.keys():
            self.add(len(data[key]['data']))

    def print_tree(self, node, level=0):
        if node is not None:
            self.print_tree(node.left, level + 1)
            print(' ' * 4 * level + '-> ' + str(node.value))
            self.print_tree(node.right, level + 1)

