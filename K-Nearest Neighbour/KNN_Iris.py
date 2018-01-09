'''
Classifier using K-Nearest Neighbour algorithm
for Iris Flower Dataset
'''
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

print(iris.feature_names)
print(iris.target_names)

for i in range(len(iris.target)):
    print("Example {}: Features: {}, Labels: {}" \
    .format(i, iris.data[i], iris.target[i]))

test_idx = [0, 50, 100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

knn = KNeighborsClassifier()
knn.fit(train_data, train_target)

print("\nTest data:\n %s" %test_data)
print("\nTest target: %s" %test_target)
print("\nPredicted target: %s\n" %knn.predict(test_data))
