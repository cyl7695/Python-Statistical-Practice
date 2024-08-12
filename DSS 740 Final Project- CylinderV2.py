# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 20:31:40 2022

@author: cyl76
"""
#Mark Cylinder Final Project 7.15.22
#Read the NFL_Data dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import pyplot
from sklearn.neighbors import KNeighborsClassifier 
   
#Import the data
NFL_Data = pd.read_csv('C:/Users/cyl76/OneDrive/Documents/My Documents/Python/2021 NFL Raw Dataset.csv')
#show the head and tail of the dataset
NFL_Data.head(10)


# print the list of all the column headers
print("The column headers are:")
print(list(NFL_Data.columns.values))
print("Shape of NFL_Data data: {}".format(NFL_Data.shape))
NFL_Data.describe().T

#We want to look into the dataset and see if there are any missing values
pd.isnull(NFL_Data).sum()

#There are no missing values and the data set looks clean and ready to go


#importing machine learning algorithms
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from IPython.display import display


#Data Exploration

#what is the win rate for the home team?
# Calculate number of features. -1 because we are saving one as the target variable (win/loss)
n_features = NFL_Data.shape[1] - 1

# Calculate number of home matches.
n_homegames = len(NFL_Data[(NFL_Data.Home_Away == 'Home')])

# Calculate matches won by home team.
n_homewins = len(NFL_Data[(NFL_Data.Win_Loss == 'Win')])

# Calculate win rate for home team.
win_rate = n_homewins/n_homegames * 100
win_rate

# Print the results
print ("Number of features: {}".format(n_features))
print ("Number of matches won by home team: {}".format(n_homewins))
print ("Win rate of home team: {:.2f}%".format(win_rate))

#Drop the Home team and Away team columns to avoid bringing in those categorical columns into the model
NFL_Data = NFL_Data.drop(['team'], 1)
NFL_Data = NFL_Data.drop(['Opponent_abbrev'], 1)
NFL_Data

#Prep the data for testing/training
X = NFL_Data.drop(['Win_Loss'],1)
y = NFL_Data['Win_Loss']

#we want continous variables that are integers for our input data, so lets remove any categorical variables
def preprocess_features(X):
    ''' Preprocesses the football data and converts catagorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)
                    
        # Collect the revised columns
        output = output.join(col_data)
    return output

X = preprocess_features(X)
print ("Processed feature columns ({} total features):\n{}".format(len(X.columns), list(X.columns)))

# Show the feature information by printing the first five rows
print ("\nFeature values:")
display(X.head())


#Train and Test the data
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

#preproccesing and transforming the data
from sklearn.preprocessing import StandardScaler
feature_scaler = StandardScaler()
X_train = feature_scaler.fit_transform(X_train)
X_test = feature_scaler.transform(X_test)
   
#Training and Evaluating Models
def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
  #Train the classifier
    clf.fit(X_train, y_train)

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
#make predictions
    y_pred = clf.predict(features)
    
    return f1_score(target, y_pred, pos_label='Win'), sum(target == y_pred) / float(len(y_pred))


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print ("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    f1, acc = predict_labels(clf, X_train, y_train)
    print (f1, acc)
    print ("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1 , acc))
    
    f1, acc = predict_labels(clf, X_test, y_test)
    print ("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1 , acc))


# Initialize the three models (XGBoost is initialized later)
clf_A = LogisticRegression(random_state = 42)
clf_B = SVC(random_state = 912, kernel='rbf')
clf_C = xgb.XGBClassifier(seed = 82)

train_predict(clf_A, X_train, y_train, X_test, y_test)
print ('')
train_predict(clf_B, X_train, y_train, X_test, y_test)
print ('')
train_predict(clf_C, X_train, y_train, X_test, y_test)
print ('')

#creating k-folds cross validation with each of the classifiers
#looking at logistic regression first
from sklearn.model_selection import cross_val_score
all_accuracies = cross_val_score(estimator=clf_A, X=X_train, y=y_train, cv=5)
print(all_accuracies)
print(all_accuracies.mean())
print(all_accuracies.std())
#The accuracies are 97.5%, 96.34%, 97.56%, 93.82%, and 95.06%
#the mean accuracies are 96.07% which is very good
#the standard deviation is 1.45% which is also very good for the model


#looking at SVC next
from sklearn.model_selection import cross_val_score
all_accuracies = cross_val_score(estimator=clf_B, X=X_train, y=y_train, cv=5)
print(all_accuracies)
print(all_accuracies.mean())
print(all_accuracies.std())
#The accuracies are 89.02%, 92.68%, 92.68%, 93.82%, and 91.35%
#the mean accuracies are 91.91% which is very good
#the standard deviation is 1.64% which is also very good for the model

#looking at XGB last
from sklearn.model_selection import cross_val_score
all_accuracies = cross_val_score(estimator=clf_C, X=X_train, y=y_train, cv=5)
print(all_accuracies)
print(all_accuracies.mean())
print(all_accuracies.std())
#The accuracies are 97.56%, 97.56%, 98.78%, 96.29%, and 100%
#the mean accuracies are 98.03% which is very good
#the standard deviation is 1.25% which is also very good for the model



#Import 'GridSearchCV' and 'make_scorer'
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


#Create the parameters list for the grid search
parameters = { 'learning_rate' : [0.1],
               'n_estimators' : [40],
               'max_depth': [3],
               'min_child_weight': [3],
               'gamma':[0.4],
               'subsample' : [0.8],
               'colsample_bytree' : [0.8],
               'scale_pos_weight' : [1],
               'reg_alpha':[1e-5]
             }  

#Looking at the XGB Classifier make an f1 scoring function using 'make_scorer' 
f1_scorer = make_scorer(f1_score,pos_label='Win')

#Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = GridSearchCV(clf_C,
                        scoring=f1_scorer,
                        param_grid=parameters,
                        cv=5)

#Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_train,y_train)

# Get the estimator
clf_C = grid_obj.best_estimator_
print (clf_C)

# Report the final F1 score for training and testing after parameter tuning
f1, acc = predict_labels(clf_C, X_train, y_train)
print ("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1 , acc))
    
f1, acc = predict_labels(clf_C, X_test, y_test)
print ("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1 , acc))




