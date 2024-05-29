import pandas as pd


class GiniImpurity:
    def __init__(self, data: pd.DataFrame, min_class_instances):
        self.data = data
        self.data_chunks = {}
        self.min_class_instances = min_class_instances

    def determine_cutting_points(self) -> dict:
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
            for index, row in sorted_data.iloc[:, attribute:attribute+1].iterrows():
                gini_points[attribute].append(row.iloc[0])
            gini_points[attribute] = list(dict.fromkeys(gini_points[attribute]))
        for key, values in gini_points.items():
            cutting_points = []
            for attribute in range(len(values)):
                if attribute == len(values) - 1:
                    continue
                cutting_points.append((values[attribute] + values[attribute + 1]) / 2)
            gini_points[key] = cutting_points
        return gini_points

    def calculate_gini_for_each_cutting_point(self, subdata):
        cutting_points_dict = self.determine_cutting_points()
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
            return 0.0

        class_column = "class" if "class" in subset.columns else subset.columns[-1]

        total_instances = len(subset)

        class_counts = subset[class_column].value_counts()
        class_distribution = class_counts / total_instances

        gini = 1 - sum(class_distribution ** 2)

        return gini

    def determine_smallest_gini(self, subdata):
        ginis = self.calculate_gini_for_each_cutting_point(subdata)
        min_gini = 1
        min_attribute = None
        min_cutting_point = None
        for attribute, cutting_point in ginis.items():
            for point, value in cutting_point.items():
                if value < min_gini:
                    min_gini = value
                    min_attribute = attribute
                    min_cutting_point = point
                else:
                    continue
        print(f"Min Gini of {min_gini} found in attribute {min_attribute} at cutting point {min_cutting_point}")
        return min_gini, min_attribute, min_cutting_point

    def split_data_recursive(self, df, parent_key='root'):
        if len(df) <= 3:
            self.data_chunks[parent_key] = {'data': df, 'gini_value': None, 'attribute': None, 'cutting_point': None}
            return

        gini_value, attribute, cutting_point = self.determine_smallest_gini(df)

        left_split = df[df.iloc[:, attribute] < cutting_point]
        right_split = df[df.iloc[:, attribute] >= cutting_point]

        if left_split.empty or right_split.empty:
            print("Empty split detected. Ending split.")
            self.data_chunks[parent_key] = {'data': df, 'gini_value': None, 'attribute': None, 'cutting_point': None}
            return

        left_key = f"{parent_key}_left"
        right_key = f"{parent_key}_right"

        if self.needs_further_splitting(left_split):
            self.split_data_recursive(left_split, left_key)
        else:
            self.data_chunks[left_key] = {'data': left_split, 'gini_value': gini_value, 'attribute': attribute,
                                          'cutting_point': cutting_point}

        if self.needs_further_splitting(right_split):
            self.split_data_recursive(right_split, right_key)
        else:
            self.data_chunks[right_key] = {'data': right_split, 'gini_value': gini_value, 'attribute': attribute,
                                           'cutting_point': cutting_point}

    def needs_further_splitting(self, df):
        class_column = "class" if "class" in df.columns else df.columns[-1]
        class_counts = df[class_column].value_counts()
        dominant_class = class_counts.idxmax()

        non_dominant_instances = class_counts.sum() - class_counts[dominant_class]
        # Todo - This might not be correct. The total number of class instances that are not part of the non dominant
        return non_dominant_instances > self.min_class_instances

    def split_data_along_cutting_point(self):
        self.split_data_recursive(self.data)


if __name__ == "__main__":
    data = pd.read_csv(
        "/home/philipp/Documents/semester_6/maschinelles_lernen/uebungen/uebung_3/iris.data",
        header=None)
    gini = GiniImpurity(data, 4)
    ginis = gini.calculate_gini_for_each_cutting_point(gini.data)
    gini.split_data_along_cutting_point()
    for chunk in gini.data_chunks.keys():
        class_column = "class" if "class" in gini.data_chunks[chunk]['data'].columns else \
        gini.data_chunks[chunk]['data'].columns[-1]
        class_counts = gini.data_chunks[chunk]['data'][class_column].value_counts()
        dominant_class_count = class_counts.max()
        remaining_class_count = class_counts.sum() - dominant_class_count
        print(chunk, len(gini.data_chunks[chunk]['data']), "Dominant Class Count:", dominant_class_count,
              "Remaining Class Count:", remaining_class_count)
    print(len(gini.data_chunks))
