from sklearn.datasets import load_iris  # Iris Datensatz importieren
from sklearn.tree import DecisionTreeClassifier  # CART-Algorithmus importieren
from sklearn.tree import plot_tree  # Funktion fuer grafische Darstellung des Baums importieren
import matplotlib.pyplot as plt  # Funktion fuer das Veraendern der Groeße des Plots importieren


iris = load_iris()

merkmale = iris.data
klasse = iris.target

modell = DecisionTreeClassifier(criterion="gini", min_impurity_decrease=0.005, min_samples_leaf=3)
# min_impurity_decrease ist der Wert, um den die Verunreinigung mindestens reduziert werden muss, damit durch den
# entsprechenden Gini-Wert ein Split des Datensatzes durchgeführt wird.
# min_sample_leaf gibt an, wie viele Zeilen unseres Datensatzes mindestens in einem Blatt vorkommen sollen.

modell.fit(merkmale, klasse)

plt.figure(figsize=(20, 20))

tree = plot_tree(modell, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, fontsize=14)

plt.show()
