# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 20:27:32 2019

@author: ashish
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pp

def costFunc(X, y, theta):
    y_pred = X.dot(theta)
    # Calculate mean squared error
    mse = ((y_pred - y) ** 2).mean()
    return mse

def scaleFeatures(X):
    min_val = X.min(axis=0)
    std_dev = X.std(axis=0)
    X_norm = (X - min_val)/ std_dev
    return X_norm

def gradientDescent(X, y, theta, alpha, iterations):
    m = X.shape[0] # number of examples
    J_history = np.zeros((iterations, 1))
    
    for i in range(iterations):
        t1 = X.dot(theta) - y
        t2 = t1 * X
        t3 = np.sum(t2, axis=0)
        #t3 = t3.reshape(t3.shape[0], 1)
        t4 = ((alpha * (t3.T))/m).reshape(2,1)
        theta = theta - t4
        J = costFunc(X, y, theta)
        J_history[i] = J
    
    return theta, J_history

# initial theta
theta = np.zeros(2).reshape(2,1)
alpha = 0.01
iterations = 1000

house_prices = pd.read_csv('house_prices.csv', header=None, 
                           names=['area', 'noOfBedrooms', 'price'])

X = house_prices['area']
X = X[:, np.newaxis]
y = house_prices['price']
y= y[:, np.newaxis]

fig, axes = pp.subplots()
axes.scatter(X, y, marker='o', color='r')
house_prices.plot.scatter(x='area', y = 'price', c='r')

# Feature Scale
X_norm = scaleFeatures(X)

# stack 1's in the first column 
X_norm = np.column_stack((np.ones(X_norm.shape[0]), X_norm))

theta = gradientDescent(X_norm, y, theta, alpha, iterations)

x = house_prices_norm['area']
x = np.column_stack((np.ones(len(x)), x))
y_act = house_prices_norm['price']
print(x.shape)
res = sp.optimize.minimize(cost_func, initial_coeff, args= (x, y_act))
res

coeff = res.x

xs = np.linspace(x.min(), x.max(), 10)
ys = np.column_stack( (np.ones(len(xs)), xs)).dot(coeff) # ys = prediction for arbitrary values of xs


axes.plot(X_norm[:, 1], X_norm[:, 1].dot(theta), color='blue')
