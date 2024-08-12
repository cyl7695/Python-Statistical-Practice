# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 17:15:19 2022

@author: cyl76
"""

#5 Using the  sklearn wine data  set, develop a decision tree classifier.  Visualize and illustrate your answer.

import sklearn
from sklearn.datasets import load_wine
import numpy as np
from  sklearn.model_selection import train_test_split
from  sklearn.model_selection import ShuffleSplit
from sklearn import tree
from sklearn import metrics

wine = load_wine()

#let's have a look at the dataset to get a basic understanding
print(wine.DESCR)
print(wine.data.shape)
print(wine.target.shape)
print(wine.feature_names)

#create the X & y variables to store the data and target values
X = wine.data
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


#create the decision tree classifer model to fit the data
tree = tree.DecisionTreeClassifier()
tree.fit(X_train, y_train)
print(tree)

#create the predicted output by passing X_Test and and expected y
expected_y  = y_test
predicted_y = tree.predict(X_test)


#print the classificaiton report and the confusion matrix for the classifier
print(metrics.classification_report(expected_y, predicted_y, target_names=wine.target_names))
print(metrics.confusion_matrix(expected_y, predicted_y))

#ANALYZING DECISION TREES
from sklearn.tree import export_graphviz
# We can visualize the tree using the export_graphviz function from the tree module
export_graphviz(tree, out_file="tree.dot", 
 impurity=False, filled=True)
#This writes the data in the .dot file - a text file for storing graphs.
#Set an option to color the nodes to majority class in each node and pass the class and 
#feature names
import graphviz
from IPython.display import display
with open("tree.dot") as f:
 dot_graph = f.read()
display(graphviz.Source(dot_graph))

#FEATURE IMPORTANCE IN TREES 
print("Feature importance: \n{}".format(tree.feature_importances_))
import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances_wine(model):
 n_features = wine.data.shape[1] 
 plt.barh(range(n_features), model.feature_importances_, align='center')
 plt.yticks(np.arange(n_features), wine.feature_names)
 plt.xlabel("Feature importance")
 plt.ylabel("Fetaure")
 plt.ylim(-1, n_features)
plot_feature_importances_wine(tree)




#6 Apply random forest model to the wine data set.  Illustrate your solution.
#Analyze Random Forest
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
import mglearn as mglearn
import matplotlib.pyplot as plt
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(X_train, y_train)
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
 ax.set_title("Tree {}".format(i))
 mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)
mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1], alpha=.4)
axes[-1, -1].set_title("Random Forest")
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
classifier = RandomForestClassifier(n_estimators = 30, max_depth=2)
classifier.fit(X_train,y_train)
prediction = classifier.predict(X_test)
accuracy = sklearn.metrics.accuracy_score(prediction,y_test)
print("Accuracy: ", '%.2f'% (accuracy*100),"%")
#We have made 6 different forest tree plots from the wine dataset using random forests from this we were able to achieve 84% accuracy


#RANDOM FOREST CONSISTING OF 100 TREES ON THE Wine DATASET
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, random_state=0)
forest=RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))
import numpy as np
import matplotlib.pyplot as plt
importances = forest.feature_importances_
features = wine['feature_names']
indices = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

#7 Develop a backpropagation neural network model for Boston Housing dataset using tensorflow library.
import pandas as pd
from sklearn.datasets import load_boston
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

Boston = load_boston()

X = pd.DataFrame(Boston['data'], columns=Boston['feature_names'])
y = pd.Series(Boston['target'])

#simple linear tensorflow. This is adding multiple layers of input data. We are giving the model its shape which is thenpassed to the dense layer.
model = Sequential()
model.add(Dense(13, input_shape=(13,), activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))

# regression - no activation function in the last layer
model.add(Dense(1))


#We will utilize the mse as the loss function
model.compile(optimizer='adam', loss='mse')

#create the training and testing model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# The Model.fit method adjusts the model parameters to minimize the loss
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

#creating a plot to compare against epoch and the mse
import matplotlib.pyplot as plt
plt.figure(figsize=(20,8))

plt.xlabel('epoch')
plt.ylabel('mse')
plt.plot(model.history.history['loss'][:])
plt.plot(model.history.history['val_loss'][:])
plt.show()

#We can see the curve is decreasing at an increasing reate initially and changes to decreasing at a decreasing rate. As Epoche increases the 
#gap between the two lines shortens until they essentially come together towards the end of the chart

from sklearn.metrics import mean_absolute_error
model.summary()

# Let's check the average and the deviation between the dataset and our model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)


print(f"We are off on average mean square average of {round(mae * 1000, 2)} US dollars. The mean price of a house in the dataset is {round(y.mean() * 1000, 2)}, while the mean of our model is {round(y_pred.mean() * 1000, 2)}")
