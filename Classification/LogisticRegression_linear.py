# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 19:49:16 2019

@author: ashish
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pp

from scipy.optimize import minimize

# Logistic Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost Function
def costFunc(theta, X, y):
    m = y.shape[0]
    z = X.dot(theta)
    h = sigmoid(z)
    term1 = y * np.log(h)
    term2 = (1- y) * np.log(1 - h)
    J = -np.sum(term1 + term2) / m
    return J

exam_scores = pd.read_csv('Exam_Scores.csv', header=None, names = ['subject 1', 'subject 2', 'admitted'])
pos = exam_scores['admitted'] == 1
neg = exam_scores['admitted'] == 0

fig, axes = pp.subplots();
axes.set_xlabel(exam_scores.columns[0])
axes.set_ylabel(exam_scores.columns[1])

axes.scatter(exam_scores.loc[pos, 'subject 1'], exam_scores.loc[pos, 'subject 2'], color='g', marker='o', label='admitted')
axes.scatter(exam_scores.loc[neg, 'subject 1'], exam_scores.loc[neg, 'subject 2'], color='r', marker='o', label='not admitted')
axes.legend()
            
X = exam_scores.iloc[:, :2]
X = np.column_stack((np.ones(X.shape[0]), X))
y = exam_scores.iloc[:, 2]

initial_theta = np.zeros(X.shape[1]).reshape(X.shape[1],1)

res = minimize(costFunc, initial_theta, args=(X,y))
theta = res.x.reshape(res.x.shape[0],1)

# to do a linear fit only need to predict highest and lower score point
x_test = np.array([X[:,1].min() - 2, X[:, 1].max() + 2])
y_test = (-1 /theta[2]) * ( (theta[1] * x_test) + theta[0] )

axes.plot(x_test, y_test, color ='black', label = 'decision boundary') 
axes.legend()