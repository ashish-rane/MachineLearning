# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 17:36:37 2019

@author: ashish
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pp
from scipy.optimize import minimize

def mapFeatures(X, degree):
    X_poly = X
    for i in range(2, degree + 1):
        X_poly = np.column_stack((X_poly, X ** i))
    return X_poly

def costfunc(theta, X, y):
    y_pred = X.dot(theta)
    mse = ((y_pred-y) ** 2).mean()
    return mse

salaries = pd.read_csv('Position_Salaries.csv')

fig, axes = pp.subplots()
axes.scatter(salaries['Level'], salaries['Salary'], marker='x', color='r')
axes.set_xticklabels(salaries['Position'])

degree = 4
X = salaries['Level'].values
X = np.column_stack((np.ones(X.shape[0]), mapFeatures(X, degree)))
y = salaries['Salary'].values


initial_theta = np.ones(degree + 1).reshape(degree + 1,1)

res = minimize(costfunc, initial_theta, args=(X,y))

theta = res.x

X_test = np.linspace(salaries['Level'].min(), salaries['Level'].max(), 10)
X_test = np.column_stack((np.ones(X_test.shape[0]), mapFeatures(X_test, degree)))

y_pred = X_test.dot(theta.reshape(theta.shape[0],1))

axes.plot(X_test[:,1], y_pred, color='blue')

