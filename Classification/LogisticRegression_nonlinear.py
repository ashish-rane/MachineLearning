# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 19:29:45 2019

@author: ashish
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pp


from scipy.optimize import minimize 


def mapFeature(X1, X2, degree):
    res = np.ones(X1.shape[0])
    for i in range(1,degree + 1):
        for j in range(0,i + 1):
            res = np.column_stack((res, (X1 ** (i-j)) * (X2 ** j)))
    
    return res

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def costFunc(theta, X, y, reg_factor):
    m = y.shape[0]
    z = X.dot(theta)
    h = sigmoid(z)
    term1 = y * np.log(h)
    term2 = (1- y) * np.log(1 - h)
    J = -np.sum(term1 + term2, axis = 0) / m
    
    # Regularization
    reg_term = (reg_factor * sum(theta ** 2))/ (2 * m)
    J = J + reg_term
    return J            


def plotDecisionBoundary(theta,degree, axes):
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    U,V = np.meshgrid(u,v)
    # convert U, V to vectors for calculating additional features
    # using vectorized implementation
    U = np.ravel(U)
    V = np.ravel(V)
    Z = np.zeros((len(u) * len(v)))
    
    X_poly = mapFeature(U, V, degree)
    Z = X_poly.dot(theta)
    
    # reshape U, V, Z back to matrix
    U = U.reshape((len(u), len(v)))
    V = V.reshape((len(u), len(v)))
    Z = Z.reshape((len(u), len(v)))
    
    cs = axes.contour(U,V,Z,levels=[0],cmap= "Greys_r", label='Decision Boundary')
    axes.legend(labels=['good', 'faulty', 'Decision Boundary'])
    return cs

### MAIN #####
degree = 6
components = pd.read_csv('component_tests.csv', header=None, names = ['feature 1', 'feature 2', 'faulty'] )

# get positive and negative samples for plotting
pos = components['faulty'] == 1
neg = components['faulty'] == 0

# Visualize Data
fig, axes = pp.subplots(nrows=1, ncols=3, figsize=(15,4));
axes1,axes2,axes3 = axes
axes1.set_xlabel('Feature 1')
axes1.set_ylabel('Feature 2')
axes1.scatter(components.loc[pos, 'feature 1'], components.loc[pos, 'feature 2'], color = 'r', marker='x', label='Faulty')
axes1.scatter(components.loc[neg, 'feature 1'], components.loc[neg, 'feature 2'], color = 'g', marker='o', label='Good')
axes1.legend(title='Legend', loc = 'best' )
axes1.set_xlim(-1,1.5)
axes1.set_xlim(-1,1.5)



X = components.iloc[:, :2]

X_poly = mapFeature(X.iloc[:, 0], X.iloc[:, 1], degree)
y = components.iloc[:, 2]

initial_theta = np.zeros(X_poly.shape[1]).reshape(X_poly.shape[1], 1)

# No Regularization
res = minimize(costFunc, initial_theta, args=(X_poly, y, 0))
theta = res.x.reshape(res.x.shape[0], 1)

# Plot Decision boundary
#fig, axes = pp.subplots();
axes2.set_xlabel('Feature 1')
axes2.set_ylabel('Feature 2')
axes2.scatter(components.loc[pos, 'feature 1'], components.loc[pos, 'feature 2'], color = 'r', marker='x', label='Faulty')
axes2.scatter(components.loc[neg, 'feature 1'], components.loc[neg, 'feature 2'], color = 'g', marker='o', label='Good')
#axes.legend(title='Legend', loc = 'best' )

plotDecisionBoundary(theta, degree, axes2)

# Apply Regularization
res = minimize(costFunc, initial_theta, args=(X_poly, y, 1))
theta = res.x.reshape(res.x.shape[0], 1)

# Plot Decision boundary
#fig, axes = pp.subplots();
axes3.set_xlabel('Feature 1')
axes3.set_ylabel('Feature 2')
axes3.scatter(components.loc[pos, 'feature 1'], components.loc[pos, 'feature 2'], color = 'r', marker='x', label='Faulty')
axes3.scatter(components.loc[neg, 'feature 1'], components.loc[neg, 'feature 2'], color = 'g', marker='o', label='Good')
#axes.legend(title='Legend', loc = 'best' )

plotDecisionBoundary(theta, degree, axes3)
