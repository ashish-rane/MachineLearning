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

def costFunc(theta, X, y):
    y_pred = X.dot(theta)
    # Calculate mean squared error
    mse = ((y_pred - y) ** 2).mean()
    J_hist.loc[len(J_hist)] = [theta, mse]
    
    
    
    global iterations 
    iterations = iterations + 1
    return mse



house_prices = pd.read_csv('house_prices.csv', header=None, 
                           names=['area', 'noOfBedrooms', 'price'])

# initial theta
initial_theta = np.ones(1).reshape(1,1)
alpha = 0.01

# features
X = house_prices['area'].values
X = X.reshape(X.shape[0], 1)

# observed predictions
y = house_prices['price'].values
y = y.reshape(y.shape[0], 1)

# Assume Theta0 = 0
# train our hypothesis to minimize cost
res = minimize(costFunc, initial_theta, args=(X, y))

theta = res.x

# Create some test data for which to predict
xs = np.linspace(house_prices['area'].min(), house_prices['area'].max(), 10)

# ys = prediction for test values of xs
ys = xs * theta

fig, axes = pp.subplots(nrows=1, ncols=2, figsize=(15,4))
ax1,ax2 = axes.flatten()
ax1.grid(color='b', alpha=0.8, linestyle='dotted', linewidth = 0.5)
ax2.grid(color='b', alpha=0.8, linestyle='dotted', linewidth = 0.5)

ax1.scatter(X, y, marker='o', color='r')
ax1.plot(xs,ys, color='green')
ax1.set_xlabel('area')
ax1.set_ylabel('price')

opacity = 0.2
# plot the progess of the algorithm
for t in J_hist['theta']:
    y = xs * t
    ax1.plot(xs, y, color='blue', alpha=opacity, linestyle='dashed')
    opacity = opacity + 0.1
    



# Plot Cost Function as function of theta
ax2.scatter(J_hist['theta'], J_hist['cost'], color='r', marker='x')
ax2.set_xlabel('theta')
ax2.set_ylabel('cost')

ax2.plot(J_hist['theta'], J_hist['cost'], marker='.', color='blue')
