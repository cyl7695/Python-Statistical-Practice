# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 20:25:26 2022

@author: cyl76
"""

#2 Develop a simple pipeline model for the Iris dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

iris_df=load_iris()

X_train,X_test,y_train,y_test=train_test_split(iris_df.data,iris_df.target,test_size=0.3,random_state=0)
pipeline_lr=Pipeline([('scalar1',StandardScaler()),('pca1',PCA(n_components=2)), ('lr_classifier',LogisticRegression(random_state=0))])
model = pipeline_lr.fit(X_train, y_train)
model.score(X_test,y_test)

#our initial accuracy of our model is 86.67%

#Import imputer and onehot
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

#filling in missing values
('imputer', SimpleImputer(strategy='most_frequent'))

#converting categorical variables
('onehot', OneHotEncoder(handle_unknown='ignore'))

#We build different pipelines for each algorithm and then we will fit to see which model performs better.

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
pipeline_lr=Pipeline([('scalar1',StandardScaler()),
                     ('pca1',PCA(n_components=2)), 
                     ('lr_classifier',LogisticRegression())])
pipeline_dt=Pipeline([('scalar2',StandardScaler()),
                     ('pca2',PCA(n_components=2)),
                     ('dt_classifier',DecisionTreeClassifier())])
pipeline_svm = Pipeline([('scalar3', StandardScaler()),
                      ('pca3', PCA(n_components=2)),
                      ('clf', svm.SVC())])
pipeline_knn=Pipeline([('scalar4',StandardScaler()),
                     ('pca4',PCA(n_components=2)),
                     ('knn_classifier',KNeighborsClassifier())])
pipelines = [pipeline_lr, pipeline_dt, pipeline_svm, pipeline_knn]
pipe_dict = {0: 'Logistic Regression', 1: 'Decision Tree', 2: 'Support Vector Machine',3:'K Nearest Neighbor'}
for pipe in pipelines:
    pipe.fit(X_train, y_train)
for i,model in enumerate(pipelines):
    print("{} Test Accuracy:{}".format(pipe_dict[i],model.score(X_test,y_test)))
    
#based off the model we can see that the best accuracy comes with the SVM function at an accuracy percentage of 93.3%



#5 Use the data on positive and negative tweets at https://github.com/lesley2958/twilio-sent-analysis to develop a sentiment analyzer.   Use the techniques illustrated in this module.  Please note that the Python file is also posted that you can use to run the analysis.

import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from urllib.request import urlopen

#we will want to use the open funciton to pull from the site and then split the file to build a list for the tweets and a list for the labels
data = []
data_labels = []
with urlopen("https://raw.githubusercontent.com/clesleycode/twilio-sent-analysis/master/pos_tweets.txt") as f:
    for i in f: 
        data.append(i) 
        data_labels.append('pos')

with urlopen("https://raw.githubusercontent.com/clesleycode/twilio-sent-analysis/master/neg_tweets.txt") as f:
    for i in f: 
        data.append(i)
        data_labels.append('neg')
        
#we want to vectorize the dataset because the data could be in any format
vectorizer = CountVectorizer(
    analyzer = 'word',
    lowercase = False,
)
features = vectorizer.fit_transform(
    data
)
features_nd = features.toarray() # for easy usage

#we will now conduct cross validation on the model using a training percentage of 80%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test  = train_test_split(
        features_nd, 
        data_labels,
        train_size=0.80, 
        random_state=42)

#we can now build the classifier for this model using logistic regression
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()

log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)

#look at the accuracy of our model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

#our model came out to an accuracy of 81.5%. while this is not the greatest it is not the worst.
