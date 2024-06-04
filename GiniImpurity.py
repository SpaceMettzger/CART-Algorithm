import pandas as pd
from BinaryTree import CARTNode


class GiniImpurity:
    def __init__(self, data: pd.DataFrame, min_leaf_size: int, min_gini_decrease: float):
        """
        @param data: Dataframe containing the data that the gini impurities are to be calculated for
        @param max_foreign_class_instances: the maximum number of class instances of the non-dominant class of each
        data_chunk
        """
        self.data = data
        self.data_chunks = []
        self.min_leaf_size = min_leaf_size
        self.min_gini_decrease = min_gini_decrease

    def determine_potential_cutting_points(self) -> dict:
        """
        Calculates the possible cutting points for all attributes by taking the unique values for each attribute,
        sorting them in ascending order, calculating the average of neighboring values, and adding the average values
        (the cutting points) to a dictionary.
        @return:
            gini_points: dictionary of the possible cutting points for all attributes
        """
        num_attributes = len(self.data.columns) - 1
        gini_points = {}
        for attribute in range(num_attributes):
            sorted_data = self.data.sort_values(by=self.data.columns[attribute])
            gini_points[attribute] = []
            # For every attribute we add the attribute index as a key to the dictionary pointing to an empty list
            for index, row in sorted_data.iloc[:, attribute:attribute+1].iterrows():
                gini_points[attribute].append(row.iloc[0])
                # Here we are adding every data point for that attribute
            gini_points[attribute] = list(dict.fromkeys(gini_points[attribute]))
            # This returns only unique values for that attribute to avoid duplicates
        for key, values in gini_points.items():
            cutting_points = []
            for attribute in range(len(values)):
                if attribute == len(values) - 1:
                    continue
                cutting_points.append((values[attribute] + values[attribute + 1]) / 2)
                # For each attribute in or gini_points dictionary, we calculate the median between each attribute and
                # its neighbour. Those are our potential cutting points.
            gini_points[key] = cutting_points
        return gini_points

    def calculate_gini_for_each_cutting_point(self, subdata: pd.DataFrame) -> dict:
        """
        For each attribute all potential cutting points are determined and the data is split in two along those points.
        For each of the split subsets, the individual gini value is calculated and used to calculate the gini impurity
        between the two subsets. The gini impurity is added to a dictionary containing each gini value for each
        cutting point for that specific attribute. That dictionary is added to another dictionary containing the gini
        values for each attribute.
        @param sub subdata: Dataframe containing the data to be split
        @return: gini_values: dict containing a dict of gini values for each potential cutting point for each attribute
        """
        cutting_points_dict = self.determine_potential_cutting_points()
        gini_values = {}

        for attribute, cutting_points in cutting_points_dict.items():
            gini_impurity = {}

            for cutting_point in cutting_points:
                mask = subdata[attribute] >= cutting_point
                df1 = subdata[mask]
                df2 = subdata[~mask]

                impurity_1 = self.calculate_single_gini_impurity(df1)
                impurity_2 = self.calculate_single_gini_impurity(df2)

                total_size = df1.shape[0] + df2.shape[0]
                if total_size > 0:
                    gini = (impurity_1 * df1.shape[0] + impurity_2 * df2.shape[0]) / total_size
                else:
                    gini = 0

                gini_impurity[cutting_point] = gini
            gini_values[attribute] = gini_impurity
        return gini_values

    @staticmethod
    def calculate_single_gini_impurity(subset: pd.DataFrame) -> float:
        """
        Calculates the Gini impurity of a given dataset. If the column with the class label is not called "class",
        the last column is assumed to contain the class label.

        @param subset: pandas DataFrame containing the data
        @return: gini: float value of the Gini impurity
        """
        if subset.empty:
            return 1.0

        class_column = "class" if "class" in subset.columns else subset.columns[-1]

        total_instances = len(subset)

        class_counts = subset[class_column].value_counts()
        class_distribution = class_counts / total_instances

        gini = 1 - sum(class_distribution ** 2)

        return gini

    def determine_smallest_gini(self, subdata):
        """
        Determines the minimum gini value and the corresponding attribute and cutting point for the data passed as an
        argument
        @param subdata: the dataset for which the smallest gini and corresponding attribute and cutting point are to be
        determined
        @return:
        min_gini: lowest gini value of the dataset
        min_attribute: attribute for which the lowest gini was calculated
        min_cutting_point: cutting point, where the gini value is lowest
        """
        ginis = self.calculate_gini_for_each_cutting_point(subdata)
        min_gini = float('inf')
        min_attribute = None
        min_cutting_point = None
        for attribute, cutting_point in ginis.items():
            for point, value in cutting_point.items():
                if value < min_gini:
                    min_gini = value
                    min_attribute = attribute
                    min_cutting_point = point

        print(f"Min Gini of {min_gini} found in attribute {min_attribute} at cutting point {min_cutting_point}")
        return min_gini, min_attribute, min_cutting_point

    def split_data_recursive(self, df, previous_cutting_point=0, previous_cutting_attribute=None, parent=None):
        if df.shape[0] < 3:
            return
        gini_value, attribute, cutting_point = self.determine_smallest_gini(df)
        base_node = CARTNode(cutting_point=cutting_point, split_point=previous_cutting_point,
                             previous_attribute=previous_cutting_attribute, attribute=attribute, gini=gini_value,
                             data=df, parent=parent)

        self.data_chunks.append(base_node)
        # Check if the dataset is small or if all attributes in the specified column are equal
        if df.shape[0] < (self.min_leaf_size * 2) or gini_value < self.min_gini_decrease:
            return base_node

        left_split = df[df.iloc[:, attribute] <= cutting_point]
        right_split = df[df.iloc[:, attribute] > cutting_point]

        if left_split.shape[0] < self.min_leaf_size or left_split.shape[0] < self.min_leaf_size:
            return

        if not left_split.empty and self.needs_further_splitting(left_split, gini_value):
            base_node.left = self.split_data_recursive(left_split, previous_cutting_point=cutting_point,
                                                       previous_cutting_attribute=attribute, parent=base_node)
        else:
            return base_node

        if not right_split.empty and self.needs_further_splitting(right_split, gini_value):
            base_node.right = self.split_data_recursive(right_split, previous_cutting_point=cutting_point,
                                                        previous_cutting_attribute=attribute, parent=base_node)
        else:
            return base_node

        return base_node

    def needs_further_splitting(self, df: pd.DataFrame, gini_value: float) -> bool:
        """
        Determines if a dataframe needs to be split further based on the max_foreign_class_instances attribute
        @param df: the dataset to be checked
        @return: boolean value
        """
        return df.shape[0] >= (self.min_leaf_size * 2) or self.min_gini_decrease < gini_value

    def split_data_along_cutting_point(self):
        """
        Function calling recursive function to split data along cutting points
        @return: None
        """
        self.split_data_recursive(self.data)


if __name__ == "__main__":
    dataset = pd.read_csv(
        "/home/philipp/Documents/semester_6/maschinelles_lernen/uebungen/uebung_3/iris.data",
        header=None)
    cart_class = GiniImpurity(dataset, 3, 0.05)
    cart_class.split_data_along_cutting_point()
    for chunk in cart_class.data_chunks:
        print(chunk)
    print(len(cart_class.data_chunks))
