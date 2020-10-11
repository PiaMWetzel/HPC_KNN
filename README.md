## KNN with Iris

**Programming Languages:** Cuda/C  
**Data:** The famous Iris data set, obtained from https://archive.ics.uci.edu/ml/datasets/iris.  
**Distance measure:** Euclidean  
**Parameters of interest:** Species of the Iris flower  
**Given Parameters:** Petal-length, petal-width, sepal-length, sepal-width  
**Algorithm:** K-Nearest Neighbors (KNN)  
**Brief Description:** Goal is to determine which Iris species (Setosa, Versicolor, or Virginica) a given test
vector belongs to by using the KNN classification algorithm. The algorithm assumes “birds of one feather
flock together” and is looking for other - already known - species whose values for petal length, petal
width, sepal length, and sepal width are most similar to our vector’s values. The similarity is determined
by the Euclidean Distance between the test and training data.

