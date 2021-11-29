# Non LinearRegression

import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('DataSets/kc_house_data.csv')
df.head(5)
X = df.drop('price', axis=1)
y = df.price
y.head(5)

import statsmodels.regression.linear_model as sm

x_opt = X
x_opt.insert(0, 'ones', 1)
x_opt.head(5)
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()
X = x_opt
X_simple_linear_regression = X

from sklearn.preprocessing import StandardScaler

simple_linear_scaler = StandardScaler()
X_simple_linear_regression = simple_linear_scaler.fit_transform(X_simple_linear_regression)
print(X_simple_linear_regression)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_simple_linear_regression, y, test_size=0.2, random_state=0)
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)
linear_model.score(x_test, y_test)

from sklearn.preprocessing import PolynomialFeatures

poly_reg_2 = PolynomialFeatures(degree=2)
X_poly_2 = poly_reg_2.fit_transform(X_simple_linear_regression)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_poly_2, y, test_size=0.2, random_state=0)
len(x_train)
len(y_train)
poly_reg = LinearRegression()
poly_reg.fit(x_train, y_train)
poly_reg.score(x_test, y_test)
poly_reg_3 = PolynomialFeatures(degree=3)
X_poly_3 = poly_reg_2.fit_transform(X_simple_linear_regression)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_poly_3, y, test_size=0.2, random_state=0)
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
lin_reg.score(x_test, y_test)

from sklearn.preprocessing import StandardScaler

sc_y = StandardScaler()
y_scaled = sc_y.fit_transform(y.values.reshape(-1, 1))
print(y_scaled)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_simple_linear_regression, y_scaled, test_size=0.2, random_state=0)

# SVR
from sklearn.svm import SVR

regressor = SVR(kernel='rbf')
regressor.fit(x_train, y_train)
regressor.score(x_test, y_test)

# Decision tree

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)
regressor.score(X_test, y_test)

# # Random Forest Regression

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_train, y_train)
regressor.score(X_test, y_test)
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X_train, y_train)
regressor.score(X_test, y_test)
