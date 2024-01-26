from sklearn.cluster import AgglomerativeClustering as agcl
from sklearn.metrics import rand_score
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt

##
##          Function definitions
##

def extract_grade(data):
    grade = []
    for i in range(len(data)):
        grade.append(data[i][0])
    data = np.delete(data, 0, 1)
    return grade, data

##
##          Algorithm
##

print("-----| Starting the algorithm |-----")
print("Reading data")

data = pd.read_csv('dataset/fashion-mnist_train.csv', header=0)
data = np.array(data[1:])

print("Downsampling the dataset")

data_res, _ = train_test_split(data, test_size=0.8, random_state=42)

print("Extracting grades")

grade, data_res = extract_grade(data_res)

print("Preparing clustering")

clustering = agcl(10, compute_full_tree = True, compute_distances = True).fit_predict(data_res)

print(rand_score(grade, clustering))
print(adjusted_rand_score(grade, clustering))

plt.scatter(data_res[:, 0], data_res[:, 1], c=clustering, cmap='viridis')
plt.title('Agglomerative Clustering - Cluster Assignments')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
#plt.show()
plt.savefig('cluster_aggl')