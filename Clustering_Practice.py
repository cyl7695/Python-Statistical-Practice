# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 19:54:32 2022

@author: cyl76
"""

#5.	Generate random data to create three clusters and illustrate three different clustering algorithms learned in this module on the generated dataset.
 
#create random random samples
import matplotlib.pyplot as plt
import numpy as np
import mglearn as mglearn
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
# generate random two-dimensional data using random state 42
X, y = make_blobs(n_samples= 100, random_state=42)

#look at X and y
X
y
#Most values of x lie between -10 to 10. All values of y lie between 0 to 2

#KMEANS using scikit learn
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
print("Cluster membership:\n{}".format(kmeans.labels_))
print("k-Means Prediction membership:\n{}".format(kmeans.predict(X)))
#here we are making the predicitons from each of the data points and classifying them as group 0,1,2.

mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')
mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1, 
2], markers='^', markeredgewidth=2)
#here we are plotting our predictions from above we can see three distinct clusters. The orange cluster 
#is bottom left hand corner the blue cluster in top middle and green cluster is on the righter middle part of the chart
#each of the groupings are about equally far distant from each other.

#AGGLOMERATIVE CLUSTERING using three clusters
from sklearn.cluster import AgglomerativeClustering
X, y = make_blobs(n_samples= 100, random_state=42)
agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X)
mglearn.discrete_scatter(X[:, 0], X[:, 1], assignment)
plt.legend(["Cluster 0", "Cluster 1", "Cluster 2"], loc="best")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
#the graph looks almost identical to the kmeans chart previously calculated
#we can see that Cluster 1 and Cluster 2 now are shown as triangles


#Using dbscan to create clustering for the random dataset
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
X, y = make_moons(n_samples=100, noise=0.05, random_state=42)

# rescale the data to zero mean and unit variance
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
dbscan = DBSCAN()
clusters = dbscan.fit_predict(X_scaled)

# plot the cluster assignments
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm2, s=60)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Cluster 0", "Cluster 1"], loc="best")
#dbscan does not allow you to manually set the amount of clusters
#instead the algorithm picks the best number of clusters based on the data
#we see that we have two distinct clusters from the random dataset. dbscan
#also goes ahead and takes away any outliers that doesnt fit with the overall trend of the data flow




#6.	Use Principal Component Analysis to illustrate dimensionality reduction on the wine data set.
from sklearn.datasets import load_wine
import mglearn as mglearn
from sklearn.model_selection import train_test_split
wine = load_wine()

#Look into the dataset
print(wine.data.shape)
print(wine.target.shape)
print(wine.feature_names)
#178 ROWS 
#14 COLUMNS

#Create the X & y variables to store the data and target values
X = wine.data
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, 
random_state=1)
print(X_train.shape) #looking at x training shape
print(X_test.shape) #looking at x test shape

#APPLYING PCA TO wine DATASET FOR VISUALIZATION
fig, axes = plt.subplots(7, 2, figsize=(10, 20))
malignant = wine.data[wine.target == 0]
benign = wine.data[wine.target == 1]
ax = axes.ravel()
for i in range(30):
 _, bins = np.histogram(wine.data[:, i], bins = 100)
 ax[i].hist(malignant[:, i], bins=bins, color=mglearn.cm3(0), alpha=.5)
 ax[i].hist(benign[:, i], bins=bins, color=mglearn.cm3(2), alpha=0.5)
 ax[i].set_title(wine.feature_names[i])
 ax[i].set_yticks(())
ax[0].set_xlabel("Fetaure magnitude")
ax[0].set_ylabel("Frequency")
ax[0].legend(["malignant", "benign"], loc="best")
fig.tight_layout()
#we can quickly see the distribution of each of the variables by using
#the first two colors to group the data. Generally the blue cluster group
#falls farther on the x axis than the green cluster group

#PCA TRANSFORMATION
scaler = StandardScaler()
scaler.fit(wine.data)
X_scaled = scaler.transform(wine.data)
from sklearn.decomposition import PCA

# keep the first two principal components of the data
pca = PCA(n_components=2)

# fit PCA model to wine data
pca.fit(X_scaled)

# transform data onto the first two principal components
X_pca = pca.transform(X_scaled)
print("Original shape: {}".format(str(X_scaled.shape)))
print("Reduced shape: {}".format(str(X_pca.shape)))
#we now are going from 13 columns to only 2 based on the number of components
#that we set for the pca

# plot first vs. second principal components, colored by class
plt.figure(figsize=(8, 8))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], wine.target)
plt.legend(wine.target_names, loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
#We can see that there are three cluster groups in the pca chart: the green cluster group,
#the orange cluster group, and the blue cluster group. We have the green cluster group bottom left hand corner with
#it slightly overlapping with the orange cluster group. The orange cluster group is top middle and it is intersectting
#slightly with both the green and blue cluster groups. The blue cluster group is far right most away from the green cluster
#group and somewhat overlapping with the orange cluster group.


#Outcome of pca transformation
print("PCA component shape: {}".format(pca.components_.shape))
print("PCA components:\n{}".format(pca.components_))
plt.matshow(pca.components_, cmap='viridis')
plt.yticks([0, 1], ["first component", "Second component"])
plt.colorbar()
plt.xticks(range(len(wine.feature_names)), wine.feature_names, rotation=60, ha='left')
plt.xlabel("Feature")
plt.ylabel("Principal components")
#here we are doing some feature extractions to look at the the most important variables
#between each components hue and diluteness is very important along with proline and alochol in the first component


#7.	Using Iris data set develop a K-Means clustering model and identify other algorithms that can improve this model.
from sklearn.cluster import KMeans 
from sklearn.datasets import load_iris
iris = load_iris()

#look into the dataset
print(iris.DESCR)
print(iris.data.shape)
print(iris.target.shape)
print(iris.feature_names)
#there are four columns that all all numerical variables.
#there is sepal length, sepal width, petal length, petal width
#there is a shape of 150 rows to 4 columns

#mean sepal length is 5.84
#mean sepal width is 3.05
#mean petal length is 3.76
#mean petal width is .76
#we will expect that sepal length and width to be larger than the respective petal length and width

X=iris['data']
y=iris['target']

#Training the data
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, 
random_state=1)
mglearn.plots.plot_scaling()

#Finding the optimum number of clusters for k-means classification to improve model
wcss = []
 
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
#Using the elbow method to determine the optimal number of clusters for k-means clustering
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.show()
#we can see a near right angle is produced at x=3 so that means the optimal
#number of k clusters to use is 3 which will be used going forward

#Building out of the clustering model
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
print("Cluster membership:\n{}".format(kmeans.labels_))
print("k-Means Prediction membership:\n{}".format(kmeans.predict(X)))
mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')
mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1, 
2], markers='^', markeredgewidth=2)

#Based on the Kmeans clustering algorithm we were able to conclude that 3 cluster groups
#were sufficient to classify the iris dataset. We can see that there are more similarities between the orange group
#and the green group as there are overlapping clusters. The first cluster group is the blue group
#located on the left most part of the graph and then the orange cluster group and lastly,
#the green cluster group on the right most part of the graph



