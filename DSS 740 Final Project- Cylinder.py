# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 19:48:49 2022

@author: cyl76
"""

#Read the NBA_Stat dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import pyplot
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
NBA_Stat = pd.read_csv('C:/Users/cyl76/OneDrive/Documents/My Documents/Python/NBA_MVP.csv', encoding= 'unicode_escape')
#show the head and tail of the dataset
NBA_Stat.head(10)
NBA_Stat.tail(10)

# print the list of all the column headers
print("The column headers are:")
print(list(NBA_Stat.columns.values))
print("Shape of NBA_Stat data: {}".format(NBA_Stat.shape))
NBA_Stat.describe().T

#We want to look into the dataset and see if there are any missing values
pd.isnull(NBA_Stat).sum()


NBA_Stat[pd.isnull(NBA_Stat["3P%"])][["Player", "3PA"]].head()


NBA_Stat[pd.isnull(NBA_Stat["FT%"])][["Player", "FTA"]].head()

NBA_Stat = NBA_Stat.fillna(0)

NBA_Stat.columns






predictors = ["Age", "G", "GS", "MP", "FG", "FGA", 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'W', 'L', 'W/L%',
       'GB', 'PS/G', 'PA/G', 'SRS']


X = NBA_Stat[~(NBA_Stat["Year"] == 2021)] #Train for Every year that is not 2021
X = NBA_Stat[~(NBA_Stat["Share"] == 0)] #filter for only players that have received shares of MVP voting
y = NBA_Stat[NBA_Stat["Year"] == 2021] #we will test for the 2021 year
X # check the MVP Candidates from 1991 - 2020



#Train and Test the data infromation
X_train, X_test, y_train, y_test = train_test_split(X, y,  
                                                    random_state=0) 

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train, y_train)



reg = Ridge(alpha=.1)
reg.fit(train[predictors],train["Share"])


predictions = reg.predict(test[predictors])
predictions = pd.DataFrame(predictions, columns=["predictions"], index=test.index)
combination = pd.concat([test[["Player", "Share"]], predictions], axis=1)
combination.sort_values("Share", ascending=False).head(20)

from sklearn.metrics import mean_squared_error

mean_squared_error(combination["Share"], combination["predictions"])
combination["Share"].value_counts()

actual = combination.sort_values("Share", ascending=False)
predicted = combination.sort_values("predictions", ascending=False)
actual["Rk"] = list(range(1,actual.shape[0]+1))
predicted["Predicted_Rk"] = list(range(1,predicted.shape[0]+1))
actual.merge(predicted, on="Player").head(5)

def find_ap(combination):
    actual = combination.sort_values("Share", ascending=False).head(5)
    predicted = combination.sort_values("predictions", ascending=False)
    ps = []
    found = 0
    seen = 1
    for index,row in predicted.iterrows():
        if row["Player"] in actual["Player"].values:
            found += 1
            ps.append(found / seen)
        seen += 1

    return sum(ps) / len(ps)
ap = find_ap(combination)
ap


years = list(range(1991,2022))
aps = []
all_predictions = []
for year in years[5:]:
    train = NBA_Stat[NBA_Stat["Year"] < year]
    test = NBA_Stat[NBA_Stat["Year"] == year]
    reg.fit(train[predictors],train["Share"])
    predictions = reg.predict(test[predictors])
    predictions = pd.DataFrame(predictions, columns=["predictions"], index=test.index)
    combination = pd.concat([test[["Player", "Share"]], predictions], axis=1)
    all_predictions.append(combination)
    aps.append(find_ap(combination))
sum(aps) / len(aps)



def add_ranks(predictions):
    predictions = predictions.sort_values("predictions", ascending=False)
    predictions["Predicted_Rk"] = list(range(1,predictions.shape[0]+1))
    predictions = predictions.sort_values("Share", ascending=False)
    predictions["Rk"] = list(range(1,predictions.shape[0]+1))
    predictions["Diff"] = (predictions["Rk"] - predictions["Predicted_Rk"])
    return predictions
add_ranks(all_predictions[1])


def backtest(NBA_Stat, model, years, predictors):
    aps = []
    all_predictions = []
    for year in years:
        train = NBA_Stat[NBA_Stat["Year"] < year]
        test = NBA_Stat[NBA_Stat["Year"] == year]
        model.fit(train[predictors],train["Share"])
        predictions = model.predict(test[predictors])
        predictions = pd.DataFrame(predictions, columns=["predictions"], index=test.index)
        combination = pd.concat([test[["Player", "Share"]], predictions], axis=1)
        combination = add_ranks(combination)
        all_predictions.append(combination)
        aps.append(find_ap(combination))
    return sum(aps) / len(aps), aps, pd.concat(all_predictions)
mean_ap, aps, all_predictions = backtest(NBA_Stat, reg, years[5:], predictors)
mean_ap



all_predictions[all_predictions["Rk"] < 5].sort_values("Diff").head(10)


pd.concat([pd.Series(reg.coef_), pd.Series(predictors)], axis=1).sort_values(0, ascending=False)

stat_ratios = NBA_Stat[["PTS", "AST", "STL", "BLK", "3P", "Year"]].groupby("Year").apply(lambda x: x/x.mean())
NBA_Stat[["PTS_R", "AST_R", "STL_R", "BLK_R", "3P_R"]] = stat_ratios[["PTS", "AST", "STL", "BLK", "3P"]]
predictors += ["PTS_R", "AST_R", "STL_R", "BLK_R", "3P_R"]
mean_ap, aps, all_predictions = backtest(NBA_Stat, reg, years[5:], predictors)
mean_ap

NBA_Stat["NPos"] = NBA_Stat["Pos"].astype("category").cat.codes
NBA_Stat["NTm"] = NBA_Stat["Tm"].astype("category").cat.codes
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=50, random_state=1, min_samples_split=5)

mean_ap, aps, all_predictions = backtest(NBA_Stat, rf, years[28:], predictors + ["NPos", "NTm"])
mean_ap

mean_ap, aps, all_predictions = backtest(NBA_Stat, reg, years[28:], predictors)
mean_ap




from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
def backtest(NBA_Stat, model, years, predictors):
    aps = []
    all_predictions = []
    for year in years:
        train = NBA_Stat[NBA_Stat["Year"] < year].copy()
        test = NBA_Stat[NBA_Stat["Year"] == year].copy()
        sc.fit(train[predictors])
        train[predictors] = sc.transform(train[predictors])
        test[predictors] = sc.transform(test[predictors])
        model.fit(train[predictors],train["Share"])
        predictions = model.predict(test[predictors])
        predictions = pd.DataFrame(predictions, columns=["predictions"], index=test.index)
        combination = pd.concat([test[["Player", "Share"]], predictions], axis=1)
        combination = add_ranks(combination)
        all_predictions.append(combination)
        aps.append(find_ap(combination))
    return sum(aps) / len(aps), aps, pd.concat(all_predictions)
mean_ap, aps, all_predictions = backtest(NBA_Stat, reg, years[28:], predictors)
mean_ap

sc.transform(NBA_Stat[predictors])



#Everything below is from past homeworks

#Create a win loss column
#df2 <- mutate(df2, win_loss = ifelse(own_score>opp_score, 'Win', 'Loss'))
#df2 <- mutate(df2, win_loss_binary = ifelse(win_loss=="Win", 1, 0))

#Create a passing % column
#df2$passing_perc <- df2$pass_completions/df2$pass_attempts
#df2$counter <- 1
#df2


#Let's look at some basic statistics to get an overall understanding of the grouped dataset before going any further with
#the analysis

#hist(df2$passing_perc, col="coral", xlab="Pass %", 
#     main="Distribution of Pass %", breaks=12,
#     xlim=c(0,1), ylim=c(0,150))


#hist(df2$interceptions, col="coral", xlab="Interceptions", 
#     main="Distribution of Interceptions", breaks=12,
#     xlim=c(0,7), ylim=c(0,300))


#hist(df2$lost_fumbles, col="coral", xlab="Lost Fumbles", 
#     main="Distribution of Lost Fumbles", breaks=12,
#     xlim=c(0,7), ylim=c(0,350))


#boxplot(df2$rush_yds, col="coral", xlab="Rush Yards", 
#        main="Distribution of Rush Yards", ylim=c(0,300))

#Cleanse the data
#We cannot have glucose, bloodpressure, skinthickness, insulin, & BMI that are 0 so we must change these to NA
NBA_Stat[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = NBA_Stat[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

#Replacing the NA with the averages
NBA_Stat['Glucose'].fillna(NBA_Stat['Glucose'].mean(), inplace = True)
NBA_Stat['BloodPressure'].fillna(NBA_Stat['BloodPressure'].mean(), inplace = True)
NBA_Stat['SkinThickness'].fillna(NBA_Stat['SkinThickness'].mean(), inplace = True)
NBA_Stat['Insulin'].fillna(NBA_Stat['Insulin'].mean(), inplace = True)
NBA_Stat['BMI'].fillna(NBA_Stat['BMI'].mean(), inplace = True)


#Look at the distributions for each variable
NBA_Stat.hist(figsize = (20,20))
plt.show()



#The target is the outcome column
#The input data is the rest of the columns except outcome
X = NBA_Stat.drop('Outcome', axis=1).values   #Input
y = NBA_Stat['Outcome'].values                #Target
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
X = NBA_Stat.drop('Outcome', axis=1).values   #Input
y = NBA_Stat['Outcome'].values                #Target
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