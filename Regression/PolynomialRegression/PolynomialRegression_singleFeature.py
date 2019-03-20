# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 17:36:37 2019

@author: ashish
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pp
from scipy.optimize import minimize

def hypothesisFunc(theta, X):
    return X.dot(theta)

def mapFeatures(X, degree):
    X_poly = X
    for i in range(2, degree + 1):
        X_poly = np.column_stack((X_poly, X ** i))
    return X_poly

def costfunc(theta, X, y):
    m = X.shape[0]
    y_pred = hypothesisFunc(theta, X)
    cost = np.sum((y_pred-y) ** 2)/(2 * m)
    return cost

salaries = pd.read_csv('Position_Salaries.csv')

fig, axes = pp.subplots()
axes.scatter(salaries['Level'], salaries['Salary'], marker='x', color='r')
axes.set_xticklabels(salaries['Position'])

degree = 5
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

