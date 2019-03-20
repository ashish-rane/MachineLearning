# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 20:12:12 2019

@author: ashish
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pp

salaries = pd.read_csv('Position_Salaries.csv')

X = salaries['Level'].values.reshape(X.shape[0], 1)
y = salaries['Salary'].values

fig, axes = pp.subplots()
axes.scatter(salaries['Level'], salaries['Salary'], marker='x', color='r')
axes.set_xticklabels(salaries['Level'].unique())

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree=3)
X = poly.fit_transform(X)

model = LinearRegression()
model.fit(X, y)

X_test = np.linspace(salaries['Level'].min(), salaries['Level'].max(), 10)
X_test = X_test.reshape(X_test.shape[0], 1)
X_test = poly.transform(X_test)

y_pred = model.predict(X_test)

axes.plot(X_test[:,1], y_pred, color='blue')