# -*- coding: utf-8 -*-
"""
Created on Mon May 30 12:39:34 2022

@author: cyl76
"""
#Problem 3
#import the statistics module
import statistics

#3a. 
#create a list of ten numbers for a dataset
numerical_dataset = [1,1,2,9,7,4,3,5,6,8]
#print the list of numbers
numerical_dataset

#3b.
#calculate the mean
calc_mean = statistics.mean(numerical_dataset)
#print the mean
print("The Mean is:", calc_mean)

#show the mode
calc_mode = statistics.mode(numerical_dataset)
#print the mode
print("The Mode is:", calc_mode)

#show the median
calc_med = statistics.median(numerical_dataset)
#print the median
print("The Median is:", calc_med)

#3c. 
#show how many times the mode appears
count_mode = numerical_dataset.count(1)
#print the mode and the number of times it appears
print("The Mode is", calc_mode, "and it appears", count_mode, "times")


#Problem 4
#import pandas
import pandas as pd

#4a.
#Read the California dataset
df = pd.read_csv('C:/Users/cyl76/OneDrive/Documents/My Documents/Python/cadataexcelCSV.csv')
#show the head and tail of the dataset
df.head(10)
df.tail(10)

# print the list of all the column headers
print("The column headers are:")
print(list(df.columns.values))

#4b. 
import matplotlib.pyplot as plt
import folium
import webbrowser

long = df[' longitude']
lat = df['latitude']

m = folium.Map(location=[lat.mean(), long.mean()], zoom_start = 6)
folium.CircleMarker(location=[lat.mean(), long.mean()], radius=50).add_to(m)
m.save("map.html")
webbrowser.open("map.html", new=2)  # open in new tab

#by marking the mean latitude and longitude we can create a center of gravity map for
#the state of california to understand the average point for the dataset. Looking at the map
#we can see that the marker is located in the south central part of the state, which
#makes sense as Los Angeles and San Diego drag down the average point downards more than the northern
#california cities such as San Jose, Oakland, San Fran, & Sacramento. 

                 

df.count()
#we have a total of 20,640 observations within our dataset
df.mean()
#average median house value is 206,901
#average median income is 3.87
#average housing median age is 28.64
#average total rooms is 2,636.25
#average total bedrooms is 537.94
#average population is 1425.80
#average households is 499.57

df.median()
#median median house value is 180,000
#median median income is 3.53
#median housing median age is 20.00
#median total rooms is 2,130.00
#median total bedrooms is 435.00
#median population is 1170.00
#median households is 409.00

# Creating box plots to show quarterlies and distribution range
med_house = df['median house value']
med_income = df['median income']
med_house_age = df[' housing median age']
tot_rooms = df['total rooms']
tot_beds = df['total bedrooms']
pop = df[' population']
households = df['households']

#median house value box plot                      
plt.boxplot(med_house)
plt.title("median house value box plot")
plt.xlabel('median house value')
plt.ylabel('distribution')
plt.show()
#The range is about 450 k with the max being 475 k and the min is about 25k. The average is around 200k. 

#median income box plot    
plt.boxplot(med_income)
plt.title("median income box plot")
plt.xlabel('median income')
plt.ylabel('distribution')
#The range is about 7, excluding outliers with the max being 8 and the min being 1. The average is around 4. 

#median house age box plot    
plt.boxplot(med_house_age)
plt.title("median house age box plot")
plt.xlabel('median house age')
plt.ylabel('distribution')
plt.show()
#The range is about 50 years with the max being about 52 years and the min is about 2 years. The average is around 30 years. 

#total rooms box plot    
plt.boxplot(tot_rooms)
plt.title("Total Rooms box plot")
plt.xlabel('Total Rooms')
plt.ylabel('distribution')
plt.show()
#The range is about 5 k excluding outliers with the max being about 5000 and the min being about 0. The average is around 2.5k. 

#total beds box plot    
plt.boxplot(tot_beds)
plt.title("Total Beds box plot")
plt.xlabel('Total Beds')
plt.ylabel('distribution')
plt.show()
#The range is about 1k excluding outliers with the max being about 1000 and the min being about 0. The average is around 500. 


#population box plot    
plt.boxplot(pop)
plt.title("Population box plot")
plt.xlabel('Population')
plt.ylabel('distribution')
plt.show()
#The range is about 3k excluding outliers with the max being about 3000 and the min being about 0. The average is around 2k. 


#households box plot    
plt.boxplot(households)
plt.title("households box plot")
plt.xlabel('Households')
plt.ylabel('distribution')
plt.show()
#The range is about 1k excluding outliers with the max being about 1000 and the min being about 0. The average is around 500.


#4c. 
# Creating historgrams to show quartiles and distribution range
import seaborn as sns

#median house value histogram   
sns.set()                    
plt.hist(med_house)
plt.title("median house value box plot")
plt.xlabel('median house value')
plt.ylabel('distribution')
plt.show()
#We see that the shape of this distribution is skewed right, which means that the mean is larger than the median. 
#the most amount of homes are valued at around 150k

#median income histogram
plt.hist(med_income)
plt.title("median income box plot")
plt.xlabel('median income')
plt.ylabel('distribution')
#We see that the shape of this distribution is skewed right, the mean is larger than the median. 
#most people have incomes around 3.

#median house age histogram
plt.hist(med_house_age)
plt.title("median house age box plot")
plt.xlabel('median house age')
plt.ylabel('distribution')
plt.show()
#Overall, the distribution looks to be skewed left but is pretty unsymmetrical as well. The mode of the distribution is about 30
#years olds with the range from 0 to 50 years old.  

#total rooms histogram
plt.hist(tot_rooms, bins=50)
plt.title("Total Rooms box plot")
plt.xlim(xmin=0, xmax = 15000)
plt.xlabel('Total Rooms')
plt.ylabel('distribution')
plt.show()
#We see that the shape of this distribution is skewed right, the mean is larger than the median. There is a much larger tail
#on this graph than the others meaning there is a pretty big spread for this variable. Mode is about 2500 rooms.

#total beds histogram
plt.hist(tot_beds, bins=50)
plt.title("Total Beds box plot")
plt.xlim(xmin=0, xmax = 2500)
plt.xlabel('Total Beds')
plt.ylabel('distribution')
plt.show()
#We see that the shape of this distribution is skewed right. Mode is about 700 beds with a range of 2000 beds


#population histogram
plt.hist(pop, bins=50)
plt.title("Population box plot")
plt.xlim(xmin=0, xmax = 7500)
plt.xlabel('Population')
plt.ylabel('distribution')
plt.show()
#We see that the shape of this distribution is skewed right. Mode is about 1000 people. The max amount of people in a residence
#is about 6000 people

#households histogram
plt.hist(households, bins=50)
plt.title("households box plot")
plt.xlim(xmin=0, xmax = 3000)
plt.xlabel('Households')
plt.ylabel('distribution')
plt.show()
#We see that the shape of this distribution is skewed right. Mode is about 400 households per residence.with a range between
#1 and 2000 households

#4d. 
#Draw a scatterplot of total rooms and median house values.

plt.scatter(tot_rooms, med_house)
plt.title('Scatterplot of total rooms and median house values')
plt.xlabel('Total Rooms')
plt.ylabel('Median House Value')
plt.show()
#We can see that there is a lot of clsuter between all of the points when comparing Total Rooms and Median House Values.
#Nearly all data points are clustered b/w 0-15000 rooms for all house values. After we go past 15000 rooms within a residence,
#we can start to see some spacing between the data points. From 15,000 room s onward there is a slight positive trend that shows
#an increase in total rooms will show an increase also in median house values. 

