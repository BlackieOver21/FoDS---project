#! python3

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import pandas as pd
from numba import jit
import matplotlib.markers

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

data = pd.read_csv('dataset/fashion-mnist_test.csv', header=0)
data = np.array(data[1:])

print("Downsampling the dataset")

data_res, _ = train_test_split(data, test_size=0.9, random_state=42)

print("Extracting grades")

grade, data_res = extract_grade(data_res)

print("Preparing PCA")

pca = PCA(n_components = 784)
reduced_data = pca.fit_transform(data_res)

print("Graphin PCA significance values")

PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()
plt.savefig("significance")

print("Preparing graph plot")

fig = plt.figure()
ax = fig.add_subplot()

clrs = ['#FF5733', '#33FF57', '#5733FF', '#FF33A1', '#33A1FF', '#A1FF33', '#FF3366', '#3366FF', '#66FF33', '#FF6633']
i = 0

print("Plotting...")

for el in grade:
    ax.scatter(reduced_data[i][0], reduced_data[i][1], c=clrs[el], marker=el)
    i+=1
    print("Element: ", i, end="\r")

annotations = [
    (matplotlib.markers.TICKLEFT),
    (matplotlib.markers.TICKRIGHT),
    (matplotlib.markers.TICKUP),
    (matplotlib.markers.TICKDOWN),
    (matplotlib.markers.CARETLEFT),
    (matplotlib.markers.CARETRIGHT),
    (matplotlib.markers.CARETUP),
    (matplotlib.markers.CARETDOWN),
    (matplotlib.markers.CARETLEFTBASE),
    (matplotlib.markers.CARETRIGHTBASE)
]

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

i=0
for anot in annotations:
    plt.text(2800, -200 - i*200, f'{anot} - ', fontsize=8, ha='right', va='bottom', color=clrs[i])
    i+=1

#plt.show()
plt.savefig('dim_red_2d')
