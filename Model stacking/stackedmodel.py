# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('machines.csv')
X = dataset.iloc[:, 2:9].values
y = dataset.iloc[:, 9:10].values

# Feature scaling of data
from sklearn.preprocessing import StandardScaler
scalerX = StandardScaler()
X = scalerX.fit_transform(X)
scalerY = StandardScaler()
y = scalerY.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
pred_lin = lin_reg.predict(X_test)

# Fitting Lasso regression
from sklearn.linear_model import Lasso
clf_lasso = Lasso(alpha = 0.1)
clf_lasso.fit(X_train, y_train)
pred_lasso = clf_lasso.predict(X_test)

# Fitting gradient boosting regressor
from sklearn.tree import DecisionTreeRegressor as treereg
clf_tree = treereg(max_depth = 4)
clf_tree.fit(X_train, y_train)
pred_tree = clf_tree.predict(X_test)

# predicted values as input to stage 2 neural network
input2 = pd.DataFrame({'softplus':pred_lin[:,0],\
            'Lasso' : pred_lasso,\
            'Decision_tree' : pred_tree,\
            'Actual': y_test[:,0]})


# Stage 2 neural network
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import ELU
regressor = Sequential()
regressor.add(Dense(output_dim = 5, init = 'uniform', input_dim = 3))
ELU(alpha=1.0)
regressor.add(Dense(output_dim = 5, init = 'uniform'))
ELU(alpha=1.0)
regressor.add(Dense(output_dim = 3, init = 'uniform'))
ELU(alpha=1.0)
regressor.add(Dense(output_dim = 1, init = 'uniform'))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
X2, y2 = input2.iloc[:,0:3].values, input2.iloc[:,3].values
regressor.fit(X2, y2, validation_split = 0.33,  batch_size = 5, nb_epoch = 100)

prednn = regressor.predict(X2)
for i,j in zip(prednn, y_test):
    print(i,j)