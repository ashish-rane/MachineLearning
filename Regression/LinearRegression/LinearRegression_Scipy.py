# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 20:27:32 2019

@author: ashish
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pp
from scipy.optimize import minimize

J_hist = pd.DataFrame(columns =['theta', 'cost'])
iterations = 0

house_prices = pd.read_csv('house_prices.csv', header=None, 
                           names=['area', 'noOfBedrooms', 'price'])

# features
X = house_prices['area'].values
X = X.reshape(X.shape[0], 1)

# observed predictions
y = house_prices['price'].values
y = y.reshape(y.shape[0], 1)


# Visualize our data
fig, axes = pp.subplots()
axes.grid(color='b', alpha=0.8, linestyle='dotted', linewidth = 0.5)
axes.scatter(X, y, marker='o', color='r')
axes.set_xlabel('area')
axes.set_ylabel('price')


# Hypothesis function - h(x) = X.theta
def hypothesisFunc(theta, X):
    return X.dot(theta)


# Cost Function - Mean Square Error
def costFunc(theta, X, y):
    m = X.shape[0]
    y_pred = hypothesisFunc(theta,X)
    y_pred = y_pred.reshape(y_pred.shape[0],1)
    
    # Calculate mean squared error
    cost = np.sum((y_pred - y) ** 2)/ (2 * m)
    J_hist.loc[len(J_hist)] = [theta, cost]
    
    
    global iterations 
    iterations = iterations + 1
    return cost



# initial theta (Assume Theta0 = 0)
initial_theta = np.ones(1).reshape((1,1))

# train our hypothesis to minimize cost
res = minimize(costFunc, initial_theta, args=(X, y))

theta = res.x.reshape((res.x.shape[0], 1))

# Create some test data for which to predict
xs = np.linspace(house_prices['area'].min(), house_prices['area'].max(), 10)
xs = xs.reshape((xs.shape[0], 1))

# ys = prediction for test values of xs
ys = hypothesisFunc(theta, xs)

# Visualize our prediction line
fig, axes = pp.subplots(nrows=1, ncols=2, figsize=(15,4))
ax1,ax2 = axes.flatten()
ax1.grid(color='b', alpha=0.8, linestyle='dotted', linewidth = 0.5)
ax2.grid(color='b', alpha=0.8, linestyle='dotted', linewidth = 0.5)

ax1.scatter(X, y, marker='o', color='r')
ax1.plot(xs,ys, color='blue')
ax1.scatter(xs,ys,color='green', s=100, marker='o')
ax1.set_xlabel('area')
ax1.set_ylabel('price')

# Visualize the progress of the algorithm
opacity = 0.2
for t in J_hist['theta']:
    y = xs * t
    ax1.plot(xs, y, color='blue', alpha=opacity, linestyle='dashed')
    opacity = opacity + 0.1
    



# Plot Cost Function as function of theta
ax2.scatter(J_hist['theta'], J_hist['cost'], color='r', marker='x')
ax2.set_xlabel('theta')
ax2.set_ylabel('cost')

ax2.plot(J_hist['theta'], J_hist['cost'], marker='.', color='blue')
