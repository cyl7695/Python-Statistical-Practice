#Problem 3 
#Practicing with Train_Test_Split
#Creating a sample
import numpy as np
from sklearn.model_selection import train_test_split
#Create a two-dimensional numpy array X and one dimensional array y
X, y = np.arange(40).reshape((10, 4)), range(10)
print("X:\n{}".format(X))
print("y:\n{}".format(list(y)))
#The input feature X and output feature y are divided into train and test sets

#The test_size is 1/5 impying that 1/5 is for the tets set and the rest 4/5 is for the 
#training set. The random-state is given 42 as seed value.
#By default shuffle is True
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42)
print("X_train:\n{}".format(X_train))
print("y_train:\n{}".format(y_train))
print("X_test:\n{}".format(X_test))
print("y_test:\n{}".format(y_test))

# Setting shuffle to False
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42, shuffle = False)
print("y_train:\n{}".format(y_train))
print("y_test:\n{}".format(y_test))


#Problem 4
#Using the Boston Housing data set visualize the data and create a linear regression model.
# Load Libraries
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Load the dataset
from sklearn.datasets import load_boston
Boston = load_boston()

#Printing scikit dictionary keys
print(Boston.data.dtype)
print(Boston.data)
print(Boston.DESCR)
print(Boston.data.shape)
print(Boston.target.shape)
print(Boston.feature_names)


#Creating the dataframe
df= pd.DataFrame(Boston.data, columns= Boston.feature_names)
df.head()
df['MEDV'] = Boston.target
df.head()
df.describe()

pyplot.figure(figsize=(12,12))
sns.heatmap(df.corr(), annot= True)

# histograms
df.hist(figsize=(12,12));

#Set the X & y variables
X = df['RM']
y = df.MEDV

plt.scatter(X, y)
plt.xlabel('total_rooms')
plt.ylabel('median_house_value')
plt.show()

#split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#look at the shape of the training and testing sets
print("X_train shape: {}".format(X_train.shape))
print("y-train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y-test shape: {}".format(y_test.shape))

#Create the linear train and testing for the linear regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

#must reshape the x train and test values in the fit and predicted
lm.fit(X_train.values.reshape(-1, 1), y_train)
y_predicted = lm.predict(X_test.values.reshape(-1, 1))

#Create the linear fit using the train set
plt.scatter(X_train.values.reshape(-1, 1), y_train, color = 'red')
plt.plot(X_train, lm.predict(X_train.values.reshape(-1, 1)), color = 'blue')
plt.xlabel('total_rooms')
plt.ylabel('median_house_value')
plt.show()

#Apply that linear fit to the test set
plt.scatter(X_test.values.reshape(-1, 1), y_test, color = 'red')
plt.plot(X_test.values.reshape(-1, 1), y_predicted, color = 'blue')
plt.title('Total Rooms vs. Median House Value (Test Set)')
plt.xlabel('total rooms')
plt.ylabel('median_house_value')
plt.show()
