#!/usr/bin/env python3

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
"""
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""
from sklearn.linear_model import LinearRegression

lin_reg= LinearRegression()
lin_reg.fit(X,y)


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4) ##change degrees to 2,3,4,5,6, see what fits best
X_poly= poly_reg.fit_transform(X)

lin_reg2= LinearRegression()
lin_reg2.fit(X_poly, y)

plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X), color='blue')
plt.title('Truth or bluff(linear regression)')
plt.xlabel("leve;")
plt.ylabel("salayt")
plt.show()



plt.scatter(X,y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Truth or bluff(poly regression)')
plt.xlabel("leve;")
plt.ylabel("salary")
plt.show()


#predict
c=[[6.5]]
a=poly_reg.fit_transform(c)
b= lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
print(" salary expected = "+str(b))
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color='blue')
plt.scatter(c,lin_reg2.predict(a), color='black')
plt.title('Truth or bluff(poly regression)')
plt.xlabel("leve;")
plt.ylabel("salary")
plt.show()
