from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import export_text
import pandas as pd
import numpy as np

def extract_grade(data):
    grade = []
    for i in range(len(data)):
        grade.append(data[i][0])
    data = np.delete(data, 0, 1)
    return grade, data

## 
##      Reading data from file
##
print("Reading data from file")

dataset_train = pd.read_csv('dataset/fashion-mnist_train.csv', header=0)
data_train = np.array(dataset_train[1:])

dataset_test = pd.read_csv('dataset/fashion-mnist_test.csv', header=0)
data_test = np.array(dataset_test[1:])

## 
##      Extracting grades
##
print("Extracting grades")

grade_train, data_train = extract_grade(data_train)
grade_test, data_test = extract_grade(data_test)

## 
##      Preparing the model
##
print("Preparing the model")


clf = DecisionTreeClassifier(random_state=42)

clf.fit(data_train, grade_train)

prediction_test = clf.predict(data_test)

accuracy = metrics.accuracy_score(prediction_test, grade_test)
print("Accuracy:", accuracy)

features = "T-shirt/top,Trouser,Pullover,Dress,Coat,Sandal,Shirt,Sneaker,Bag,Ankle boot"

# tree_rules = export_text(clf)
# print("Decision Tree Rules:\n", tree_rules)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
plot_tree(clf, filled=True, rounded=True)
plt.show()