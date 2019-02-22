# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 20:27:32 2019

@author: ashish
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pp


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
    #J_hist.loc[len(J_hist)] = [theta, cost]
    
    
    #global iterations 
    #iterations = iterations + 1
    return cost

def gradientDesc(num_iter, theta, X, y, alpha):
    m = X.shape[0]
    J_hist = np.zeros((num_iter, 2))
    for i in range(num_iter):
        prediction = hypothesisFunc(theta, X)
        error = prediction - y
        theta = theta - (alpha * (1/m) * np.sum(X * error))
        cost = costFunc(theta, X, y)
        J_hist[i, 0] = theta[0]
        J_hist[i, 1] = cost
    
    return (J_hist, theta)

# f is a series or array
class MinMaxScaler:
    def __init__(self):
        self.mean = 0
        self.variance = 0
        
    def scaleFeatures(self, f):
        self.variance = np.max(f) - np.min(f)
        self.mean = np.mean(f)
        return (f - self.mean)/ self.variance
        
    def inverseScaleFeatures(self, f):
        return (f * self.variance) + self.mean     

# initial theta (Assume Theta0 = 0)
initial_theta = np.ones(1).reshape((1,1))
alpha = 0.01

scaler_X = MinMaxScaler()
X = scaler_X.scaleFeatures(X)
scaler_Y = MinMaxScaler()
y = scaler_Y.scaleFeatures(y)

# train our hypothesis to minimize cost
J_hist, theta = gradientDesc(500, initial_theta, X, y, alpha)


# Create some test data for which to predict
xs = np.linspace(house_prices['area'].min(), house_prices['area'].max(), 10)
xs = xs.reshape((xs.shape[0], 1))
xs = scaler_X.scaleFeatures(xs)

# ys = prediction for test values of xs
ys = hypothesisFunc(theta, xs)

# Inverse transform
xs = scaler_X.inverseScaleFeatures(xs)
ys = scaler_Y.inverseScaleFeatures(ys)
X = scaler_X.inverseScaleFeatures(X)
y = scaler_Y.inverseScaleFeatures(y)


# Visualize our prediction line
fig,ax = pp.subplots()
ax.grid(color='b', alpha=0.8, linestyle='dotted', linewidth = 0.5)

ax.scatter(X, y, marker='o', color='r')
ax.plot(xs,ys, color='blue')
ax.scatter(xs,ys,color='green', s=100, marker='o')
ax.set_xlabel('area')
ax.set_ylabel('price')
