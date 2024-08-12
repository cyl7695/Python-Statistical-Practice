# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 15:22:04 2022

@author: cyl76
"""
#PROBLEM 4
#Read the Diabetes dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import pyplot
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
diabetes = pd.read_csv('C:/Users/cyl76/OneDrive/Documents/My Documents/Python/diabetes.csv')
#show the head and tail of the dataset
diabetes.head(10)
diabetes.tail(10)

# print the list of all the column headers
print("The column headers are:")
print(list(diabetes.columns.values))
print("Shape of Biabetes data: {}".format(diabetes.shape))
diabetes.describe().T
diabetes.min().T

#Cleanse the data
#We cannot have glucose, bloodpressure, skinthickness, insulin, & BMI that are 0 so we must change these to NA
diabetes[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

#Replacing the NA with the averages
diabetes['Glucose'].fillna(diabetes['Glucose'].mean(), inplace = True)
diabetes['BloodPressure'].fillna(diabetes['BloodPressure'].mean(), inplace = True)
diabetes['SkinThickness'].fillna(diabetes['SkinThickness'].mean(), inplace = True)
diabetes['Insulin'].fillna(diabetes['Insulin'].mean(), inplace = True)
diabetes['BMI'].fillna(diabetes['BMI'].mean(), inplace = True)


#Look at the distributions for each variable
diabetes.hist(figsize = (20,20))
plt.show()

#looking at the correlation between each variable
diabetes.corr()
pyplot.figure(figsize=(12,12))
sns.heatmap(diabetes.corr(), annot= True)


#The target is the outcome column
#The input data is the rest of the columns except outcome
X = diabetes.drop('Outcome', axis=1).values   #Input
y = diabetes['Outcome'].values                #Target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# Create a k-NN classifier with 3 neighbors
knn = KNeighborsClassifier(n_neighbors = 3)
#Fit the classifier to the training data
knn.fit(X_train, y_train)
print("test set predictions: {}".format(knn.predict(X_test)))
print("test set accuracy: {:.2f}".format(knn.score(X_test, y_test)))

#plot
from sklearn.datasets import make_blobs
X, y = make_blobs(centers=2, random_state=42)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel("Fetaure 0")
plt.ylabel("Fetaure 1")
plt.legend(['Class 0', 'Class 1'])
plt.show()

#Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))


#Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    #Fit the classifier to the training data
    knn.fit(X_train, y_train)
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test)
    
#Generate plot
plt.title('KNN varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()    
    

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)
# Generate the confusion matrix and classification report
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))

#Looking ath the confusion matrix we predicted 130 positives and 62 negatives
#what we got was 149 positives with 43 negatives.

#We have 117 True positives and 30 True negatives, so we predicted 147 correct
#We have 13 False positives and 32 False negatives, so we predicted 45 inccorect
#147/(147+45) 
#Giving us an accuracy of 76.5%


#PROBLEM 5 SVM Model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.metrics import accuracy_score

#Splitting the dataset into training and testing sets.
X = diabetes.drop('Outcome', axis=1).values   #Input
y = diabetes['Outcome'].values                #Target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Creating the SVM model.
clf = svm.SVC(kernel='rbf')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

#Showing accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))

#Looking ath the confusion matrix we predicted 130 positives and 62 negatives
#what we got was 150 positives with 42 negatives.

#We have 118 True positives and 30 True negatives, so we predicted 148 correct
#We have 12 False positives and 32 False negatives, so we predicted 44 inccorect
#148/(148+44) 
#Giving us an accuracy of 77.1%

#This SVM model is slightly better than the knn model by .3%


