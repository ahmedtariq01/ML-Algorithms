# Get/Download Dataset
# Import Libraries
# Import DataSets Set into memory objects
# Fill/Handle Missing DataSets (if there is missing data)
# Make the data Categorical --> converting text values to numeric -- categorical 1,2,3 -OR- one hot encoding
# Split data into Training and Test sets --> X_train, X_test, y_train, y_test
# Feature Scaling --> bringing all features values in comparable ranges say (0,1)
# DataSets Processing --> Good to go for data processing

import numpy as np
import pandas as pd

dataset = pd.read_csv('DataSets/heart.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 13].values

print('Displaying DataSets for X \n', X)
print('\n Displaying DataSets for y \n', y)

# dealing with the missing values
# Imputer class has been deprecated in newer versions so better use SimpleImputer
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# axis 0 means mean of column, axi =1 means mean of row
# strategy mean, median, most_frequent
imputer = imputer.fit(X[:, 1:13])
X[:, 1:13] = imputer.transform(X[:, 1:13])
print('Filling the value using Mean method: \n')
print('Displaying DataSets for X \n', X)
print('\n Displaying DataSets for y \n', y)

imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer = imputer.fit(X[:, 1:13])
print('\n Splitting the Values In X \n')
print('Displaying DataSets for X \n', X)
X[:, 1:13] = imputer.transform(X[:, 1:13])
print('Filling the value using Median method: \n')
print('Displaying DataSets for X \n', X)
print('\n Displaying DataSets for y \n', y)

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer = imputer.fit(X[:, 1:13])
print(' Splitting the Values In X \n')
print(' Displaying DataSets for X \n', X)
X[:, 1:13] = imputer.transform(X[:, 1:13])
print('Filling the value using Most Frequent method: \n')
print(' Displaying DataSets for X \n', X)
print('\n Displaying DataSets for y \n', y)

# Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])
print('\n Displaying DataSets in X after Encoding the first column \n', X)

# Thus, we should also use OneHotEncoding by adding dummy columns as per number of distinct values in column country
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)
print('\n DataSets After Encoding and Sorting the column \n')
print('Displaying DataSets for X \n', X)
print('\n Displaying DataSets for y \n', y)


# Splitting data set into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# X for independent variables and y for dependent variables
# random_state for random sampling
# we can also use stratified sampling (almost equal number of representation for each class)


# Feature Scaling What is feature scaling Why is it important How is it done theoretically how can we implement it in
# Python Age is between 20s and 50s, whereas salary is between 40K - 80K In this case salary becomes overwhelming
# attribute, Euclidean distance will be computed by salary with a little effect of age x_stand= x - mean(
# x)/standard_deviation(x) 44 - 37.6/4.1 OR x_norm  = x - min(x)/max(x)- min(x) max(x) --> 1 min(x) --> 0

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
print('\n Training data in X  \n ', X_train)
X_train = sc_X.fit_transform(X_train)
print('\n Training data in X  \n ', X_train)
print('\n Testing data in X  \n ', X_test)
X_test = sc_X.transform(X_test)
print('\n Testing data in X  \n ', X_test)
# for classification problems we do not need to apply feature scaling on test variable, however for regression we
# should apply it
