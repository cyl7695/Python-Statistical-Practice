# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 21:14:34 2022

@author: cyl76
"""

#4 How would you choose the right metrics for machine learning models?   Explain.  Give three models and supporting techniques.
from sklearn.datasets import load_digits 
import matplotlib.pyplot as plt 
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix 
from sklearn.linear_model import LogisticRegression 
  

#Confusion Matrix Accuracy
digits = load_digits() 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split( 
        digits.data, digits.target, random_state=0) 
lr = LogisticRegression().fit(X_train, y_train) 
pred=lr.predict(X_test) 
print("Accuracy: {:.3f}".format(accuracy_score(y_test, pred))) 
print("Confusion matrix:\n{}".format(confusion_matrix(y_test, pred))) 
 
#F1 Score
from sklearn.metrics import classification_report 
print(classification_report(y_test, pred)) 
 
from sklearn.metrics import f1_score 
print("Micro average f1 score: {:.3f}".format( 
        f1_score(y_test, pred, average="micro"))) 
print("Macro average f1 score: {:.3f}".format( 
        f1_score(y_test, pred, average="macro"))) 
 

#AUC Metric Evaluations
from sklearn.model_selection import cross_val_score 
# default scoring for classification accuracy 
print("Default scoring: {}".format( 
            cross_val_score(SVC(), digits.data, digits.target == 9))) 
# providing scoring = "accuracy" doesn't change the results 
explicit_accuracy = cross_val_score(SVC(), digits.data, digits.target == 9, 
                                    scoring="accuracy") 
print("Explicit accuracy scoring: {}".format(explicit_accuracy)) 
roc_auc = cross_val_score(SVC(), digits.data, digits.target == 9, 
                                    scoring="roc_auc") 
print("AUC accuracy scoring: {}".format(roc_auc)) 


#5 Using the wine data set use k-fold cross-validation for a random forest classifier model.  Preprocess the data and apply suitable transformation.
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
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

#preproccesing and transforming the data
feature_scaler = StandardScaler()
X_train = feature_scaler.fit_transform(X_train)
X_test = feature_scaler.transform(X_test)

#Training and cross valdation
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=300, random_state=0)


from sklearn.model_selection import cross_val_score
all_accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=5)
print(all_accuracies)
print(all_accuracies.mean())
print(all_accuracies.std())
#The accuracies are 100%, 100%, 92.59%, 96.15%, and 100%
#the mean accuracies are 97.7% which is very good
#the standard deviation is 2.97% which is also very good for the model




#6.	Use grid search method to find the best model for the wine dataset.
#The parameter values that we want to try out are passed in the list. 
from sklearn.svm import SVC 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.datasets import load_wine 
wine = load_wine() 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target,  
                                                    random_state=0) 
print("Size of training set: {}".format( 
        X_train.shape[0], X_test.shape[0])) 
best_score=0 
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]: 
    for C in [0.001, 0.01, 0.1, 1, 10, 100]: 
        # for each combination of parameters, train an SVC 
        svm = SVC(gamma=gamma, C=C) 
        svm.fit(X_train, y_train) 
        # evaluate the SVC on the test set) 
        score = svm.score(X_test, y_test) 
        # if we got a better score, stor ethe score and parameters 
        if score > best_score: 
            best_score = score 
            best_parameters = {'C':C, 'gamma':gamma} 
 
print("Best score: {:.2f}".format(best_score)) 
print("best parameters: {}".format(best_parameters)) 

#based on the bsst parameters chosen from the grid search we found an accuracy rating of 84%
#we get a training set size of 133 datapoints and found that the best C size is 10 and the
#best gamma size is .001