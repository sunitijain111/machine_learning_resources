
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y=y.reshape(-1,1)  ## important do (-1,1) not (1,-1)
y = sc_y.fit_transform(y)

#svr regresor
from sklearn.svm import SVR



reg= SVR() #kernel as poly, linear, gaussian  rbf
reg.fit(X,y)

y_pred= reg.predict(sc_X.transform(np.array([6.5]).reshape(1,-1)))
##feature scale it too. inverse trnasform it too.
y_pred= sc_y.inverse_transform(y_pred)


# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, reg.predict(X), color = 'blue')
plt.scatter(sc_X.fit_transform([[6.5]]), reg.predict(sc_X.fit_transform([[6.5]])), color="black")
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
