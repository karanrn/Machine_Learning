# Ridge Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('~/Downloads/china_carbon.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Polynomial Regression with Ridge regression (Regularistion)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree =4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
Ridge_poly = Ridge(alpha = 0)
Ridge_poly.fit(X_poly,y)
print("Score with alpha 1 : {}".format(Ridge_poly.score(X_poly,y)))