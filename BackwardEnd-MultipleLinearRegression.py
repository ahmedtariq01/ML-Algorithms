# Multiple Linear Regression without any elimination

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

# Importing the dataset
dataset = pd.read_csv('DataSets/Real Estate.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

##Apply Backward elimination to achieve optimal model
import statsmodels.formula.api as sm

# add a feature to the matrix X that carries value 1
# this 1 is actually the value of x_0 variable for b_0 the y intercept.
# X = np.append(arr = X, values = np.ones((50,1)).astype(int), axis = 1)
# This will add the 1's column to the end of the matrix, but we want it added as the first column
# so we swap the existing matrix with 1's column in the append fucntion
print(X)
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
print(X)

## For backward elimination we need to create another optimal matrix and initiate with matrix X
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
print(X_opt)
print(y)

# The regressor that supports backword elimination comes through the statsmodel api
import statsmodels.api as sm

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 2, 3, 4, 5]]
import statsmodels.api as sm

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 2, 4, 5]]
import statsmodels.api as sm

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt
