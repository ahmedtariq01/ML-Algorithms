import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Data/Pumpkin Price Data.csv')
df.head(10)
scatter_x = df.price
scatter_y = df.sizee
plt.scatter(scatter_y, scatter_x)
plt.show()
X = df.drop('price', axis=1)
y = df.price

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X = sc_X.fit_transform(X)
print(X)

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)

print(X_poly)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=0)
print(x_train)
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title("Truth or Bluff(Polynomial)")
plt.xlabel('price')
plt.ylabel('size')
plt.show()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(x_train, y_train)

# Visualising the Linear Regression results
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, model.predict(x_train), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('price')
plt.ylabel('size')
plt.show()

model.score(x_test, y_test)
