import numpy as np

from sklearn.cluster import MeanShift, estimate_bandwidth

from sklearn.decomposition import PCA
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

print("Preparing PCA")

pca = PCA(n_components = 784)
reduced_data = pca.fit_transform(data_res)
reduced_data = reduced_data[:, :10]



bandwidth = estimate_bandwidth(reduced_data, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(reduced_data)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

print(rand_score(grade, labels))
print(adjusted_rand_score(grade, labels))

import matplotlib.pyplot as plt

plt.figure(1)
plt.clf()

colors = ["#dede00", "#377eb8", "#f781bf"]
markers = ["x", "o", "^"]

for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(reduced_data[my_members, 0], reduced_data[my_members, 1], markers[k], color=col)
    plt.plot(
        cluster_center[0],
        cluster_center[1],
        markers[k],
        markerfacecolor=col,
        markeredgecolor="k",
        markersize=14,
    )

plt.title("Estimated number of clusters: %d" % n_clusters_)
#plt.show()
plt.savefig("cluster_mean")