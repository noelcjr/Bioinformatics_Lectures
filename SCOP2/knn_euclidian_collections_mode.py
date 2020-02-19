# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 06:20:01 2016

@author: noel
"""
'''
EXERCISE: 
1) "Human Learning" with iris data. This excersice 
    gives the results of clasifying with arbitrary human parameters.
2) A KNN algorithm 
    
Note: this code was thought as part of a lecture at Counter Culture Labs on 2/18/2020
'''
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

''' Human Learning: '''
# load the famous iris data
iris = load_iris()

# what do you think these attributes represent?
iris.data
iris.data.shape
iris.feature_names
iris.target
iris.target_names
# intro to numpy
type(iris.data)
## PART 1: Read data into pandas and explore

iris.feature_names
# the feature_names are a bit messy, let's 
# clean them up. remove the (cm)
# at the end and replace any spaces with an underscore
# create a list called "features" that 
# holds the cleaned column names
features = [i.replace(' ','_')[:-5] for i in iris.feature_names]
features
# read the iris data into pandas, with our refined column names
df = pd.DataFrame(iris.data, columns=features)
df.head()
# create a list of species (should be 150 elements) 
# using iris.target and iris.target_names
# resulting list should only have the words "setosa", "versicolor", and "virginica"
'''
species ==  
['setosa',
 'setosa',
 'setosa',
 'setosa',
...
...
 'virginica',
 'virginica']

Hint: use the iris.target_names and iris.target arrays
'''
species = [iris.target_names[i] for i in iris.target]
# add the species list as a new DataFrame column4
df['species'] = species

# explore data numerically, looking for differences between species
# try grouping by species and check out the different predictors
# explore data numerically, looking for differences between species
df.describe()
df.groupby('species').sepal_length.mean()
df.groupby('species')['sepal_length', 'sepal_width', 'petal_length', 'petal_width'].mean()
df.groupby('species').agg(np.mean)
df.groupby('species').agg([np.min, np.max])
df.groupby('species').describe()
'''
agg is a new function we haven't seen yet. It will
aggregate each column using specified lists of functions.
We have been using some of its shortcuts but using
agg allows us to put in many functions at a time

df.groupby('species').agg(np.mean)
==
df.groupby('species').mean()

BUT 
df.groupby('species').agg([np.min, np.max])

doesn't have a short form
'''
# explore data by sorting, looking for differences between species
df.sort_index(by='sepal_length').values
df.sort_index(by='sepal_width').values
df.sort_index(by='petal_length').values
df.sort_index(by='petal_width').values

# I used values in order to see all of the data at once
# without .values, a datafram/home/noelcjr/Documents/GA_Assignments/datae is returned

## PART 2: Write a function to predict the species for each observation

'''Create index to reference columns by name!!!'''
# create a dictionary so we can reference columns by name
# the key of the dictionary should be the species name
# the values should be the the strings index in the columns
# col_ix['sepal_length'] == 0
# col_ix['species'] == 4
col_ix = {col:index for index, col in enumerate(df.columns)}

# define function that takes in a row of data and returns a predicted species
def classify_iris(data):
    if data[col_ix['petal_length']] < 3:
        return 'setosa'
    elif data[col_ix['petal_width']] < 1.8:
        return 'versicolor'
    else:
        return 'virginica'

# make predictions and store as numpy array
preds = np.array([classify_iris(row) for row in df.values])

# calculate the accuracy of the predictions
np.mean(preds == df.species.values)

''' KNN algorithm '''
'''
Our KNN algorithm
'''
# imports go here
import pandas as pd
import numpy as np
'''
Part 1: Setting it all up

'''
# we will need a euclidean_distance_algorithm that takes in
# two numpy arrays, and calculates the 
# euclidean distance between them

def euclidean_distance(np1, np2):
    return np.linalg.norm(np1-np2)
'''
Bring in the iris data from the web
iris_data ==
    2D numpy array of the four predictors of iris
        plus the species
'''
iris_data = pd.read_csv('/home/noel/Projects/Bioinformatics/SCOP2/files/iris.csv')
# iris_data is a dataframe, but let's turn it into
# a 2D numpy array
# Hint: use .values to turn a dataframe into a 2d array

iris_data = iris_data.values

# Question: in terms of machine learning:
#   a. the first four columns are called what?
#   b. the species column is called what?

iris_data
'''
array([[5.1, 3.5, 1.4, 0.2, 'Iris-setosa'],
       [4.9, 3.0, 1.4, 0.2, 'Iris-setosa'],
       [4.7, 3.2, 1.3, 0.2, 'Iris-setosa'],
       ... ...
       [6.2, 3.4, 5.4, 2.3, 'Iris-virginica'],
       [5.9, 3.0, 5.1, 1.8, 'Iris-virginica']], dtype=object)
'''  
'''
Part 2: Predictions

Before we jump into making a general function,
let's try to predict 

unknown = [ 6.3,  3.1 ,  5.1,  2.4] with 3 neighbors
'''
# define our variables
unknown = [ 6.3,  3.1 ,  5.1,  2.4]
k = 3

# Make a a list "distances" consisting of tuples
# Each tuple should be
# (euc_distance(unknown, data_point), species)
# for each data_point in iris_data
distances = [(euclidean_distance(unknown, row[:-1]),row[-1]) for row in iris_data]
# OR 
distances = []
for row in iris_data:
    flower_data = row[:-1]
    distance = euclidean_distance(unknown, flower_data)
    distances.append((distance, row[-1]))

distances
'''
== [(4.4866468548349108, 'setosa'),
 (4.5276925690687078, 'setosa'),
 (4.6743983570080969, 'setosa'),
 ...
 (0.44721359549995821, 'virginica'),
 (0.72801098892805138, 'virginica')]
'''
# Grab the nearest k neighbors
# Now we need to take the most frequently occuring flower
# in those k neighbors
# To do this, we will use the collections module
# given a list l, this code will spit back the mode
from collections import Counter
l = [1,2,3,4,3,2,2,5,8,2,2,2,5,9,2,2,5,5,3,2]
Counter(l).most_common(1)[0][0] # == 2

# use it to find the most frequent occuring flower in nearest
# note that the species is in the 1th index
''' Nearest is not defined yet, but it is in the function below'''
prediction = Counter([n[1] for n in nearest]).most_common(1)[0][0]
'''
Time to put it in a function so we 
can apply the prediction
to each row in our data set!

most_common([n])
Return a list of the n most common elements and their counts from the most 
common to the least. If n is omitted or None, most_common() returns all 
elements in the counter. 
Elements with equal counts are ordered arbitrarily:
'''
    
# will default to 3 neighbors
def predict(unknown, k = 3):
    '''
    Input:
        unknown  == four attributes of an unknown flower
        k        == the number of neighbors used
    Output:
        A prediction of the species of flower (str)
    '''
    distances = [(euclidean_distance(unknown, row[:-1]),row[-1]) for row in iris_data]
    nearest = sorted(distances)[:k]
    #print(nearest)

    return Counter([n[1] for n in nearest]).most_common(1)[0][0]
    
predict([ 5.8,  2.7,  5.1,  1.9]) # == 'virginica'

'''Putting it all together'''
# predict each row
# Note I use row[-1] because the last element of each row 
# is the actual species
predictions = np.array([predict(row[:4]) for row in iris_data])
# this is the last column of the iris_data
actual = np.array([row[-1] for row in iris_data])
# accuracy of the model
np.mean(predictions == actual)
# now with k == 5
predictions = np.array([predict(row[:4], k = 5) for row in iris_data])
# this is the last column of the iris_data
actual = np.array([row[-1] for row in iris_data])
# accuracy of the model
np.mean(predictions == actual)
# now with k == 2
predictions = np.array([predict(row[:4], k = 2) for row in iris_data])
# this is the last column of the iris_data
actual = np.array([row[-1] for row in iris_data])
# accuracy of the model
np.mean(predictions == actual)
# now with k == 1
predictions = np.array([predict(row[:4], k = 1) for row in iris_data])
# this is the last column of the iris_data
actual = np.array([row[-1] for row in iris_data])
# accuracy of the model
np.mean(predictions == actual)
# only two neighbors is the best so far!
