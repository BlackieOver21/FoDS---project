from decision_tree import *

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

data_train = pd.read_csv('dataset/fashion-mnist_train.csv', header=0)
data_train = np.array(data_train[1:])

data_test = pd.read_csv('dataset/fashion-mnist_test.csv', header=0)
data_test = np.array(data_test[1:])

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

grade_train = np.reshape(grade_train, (59999, 1))

classifier = DecisionTreeClassifier(min_samples_split=7, max_depth=20)

classifier.fit(data_train,grade_train)

classifier.print_tree()

## 
##      Making predictions
##
print("Making predictions")

grade_predict = classifier.predict(data_test) 

from sklearn.metrics import accuracy_score

accuracy_score(grade_test, grade_predict)
